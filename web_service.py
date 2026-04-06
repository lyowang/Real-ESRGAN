"""
Real-ESRGAN Web Service
=======================
A Flask-based HTTP API that wraps Real-ESRGAN's super-resolution inference,
allowing remote clients to upscale images and videos via HTTP requests.

Usage:
    python web_service.py [--host 0.0.0.0] [--port 8288] [--gpu-id 0] [--fp32]

Endpoints:
    GET  /health          Health check
    GET  /models          List available models
    POST /upscale         Upscale a single image
    POST /upscale/video   Upscale a video file
"""

import argparse
import cv2
import logging
import numpy as np
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid

import torch
from flask import Flask, request, jsonify, send_file

# ---------------------------------------------------------------------------
# Compatibility shim: newer torchvision removed `transforms.functional_tensor`.
# basicsr still imports from it, so we create a shim module.
# ---------------------------------------------------------------------------
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import types
    import torchvision.transforms.functional as _F
    _shim = types.ModuleType('torchvision.transforms.functional_tensor')
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _shim

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

logger = logging.getLogger('realesrgan_web')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)

# Thread lock to serialise GPU inference (one inference at a time)
_inference_lock = threading.Lock()

# Cache: (model_name, tile, half) -> RealESRGANer instance
_model_cache: dict = {}

# Will be set at startup from CLI args
_gpu_id: int | None = None
_use_fp32: bool = False

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    'RealESRGAN_x4plus': {
        'description': 'General-purpose x4 upscaler (RRDB, 23 blocks). Best quality for photos.',
        'build': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=23, num_grow_ch=32, scale=4),
        'netscale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        ],
    },
    'RealESRNet_x4plus': {
        'description': 'General-purpose x4 upscaler (RRDB, 23 blocks). PSNR-oriented.',
        'build': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=23, num_grow_ch=32, scale=4),
        'netscale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
        ],
    },
    'RealESRGAN_x4plus_anime_6B': {
        'description': 'Optimised for anime images. Smaller model (6 blocks).',
        'build': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=6, num_grow_ch=32, scale=4),
        'netscale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        ],
    },
    'RealESRGAN_x2plus': {
        'description': 'General-purpose x2 upscaler (RRDB, 23 blocks).',
        'build': lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=23, num_grow_ch=32, scale=2),
        'netscale': 2,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        ],
    },
    'realesr-animevideov3': {
        'description': 'Compact model for anime video frames (VGG-style, 16 convs).',
        'build': lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                         num_conv=16, upscale=4, act_type='prelu'),
        'netscale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        ],
    },
    'realesr-general-x4v3': {
        'description': 'Compact general model with denoising control (VGG-style, 32 convs).',
        'build': lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                         num_conv=32, upscale=4, act_type='prelu'),
        'netscale': 4,
        'urls': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ],
    },
}

VALID_MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_path(model_name: str) -> str:
    """Return a local path to the model weights, downloading if necessary."""
    info = MODEL_REGISTRY[model_name]
    model_path = os.path.join(ROOT_DIR, 'weights', f'{model_name}.pth')
    if not os.path.isfile(model_path):
        for url in info['urls']:
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, 'weights'),
                progress=True,
                file_name=None,
            )
    return model_path


def get_upsampler(
    model_name: str,
    tile: int = 0,
    denoise_strength: float = 0.5,
) -> RealESRGANer:
    """Build or retrieve a cached RealESRGANer for the given model."""
    half = not _use_fp32
    cache_key = (model_name, tile, half)

    if cache_key in _model_cache:
        logger.info('Cache hit for model %s (tile=%d, half=%s)', model_name, tile, half)
        return _model_cache[cache_key]

    logger.info('Loading model %s (tile=%d, half=%s) ...', model_name, tile, half)
    info = MODEL_REGISTRY[model_name]
    model = info['build']()
    netscale = info['netscale']
    model_path = _resolve_model_path(model_name)

    # DNI (deep network interpolation) for realesr-general-x4v3
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=_gpu_id,
    )
    _model_cache[cache_key] = upsampler
    logger.info('Model %s loaded successfully.', model_name)
    return upsampler


def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in ('true', '1', 'yes')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_info = {}
    if torch.cuda.is_available():
        cuda_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'memory_allocated_mb': round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
            'memory_reserved_mb': round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
        }
    return jsonify({
        'status': 'ok',
        'device': device,
        'cuda': cuda_info,
        'models_loaded': list(_model_cache.keys()),
        'fp32_mode': _use_fp32,
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models and their descriptions."""
    models = []
    for name, info in MODEL_REGISTRY.items():
        weight_path = os.path.join(ROOT_DIR, 'weights', f'{name}.pth')
        models.append({
            'name': name,
            'description': info['description'],
            'scale': info['netscale'],
            'weights_downloaded': os.path.isfile(weight_path),
        })
    return jsonify({'models': models})


@app.route('/upscale', methods=['POST'])
def upscale_image():
    """
    Upscale a single image.

    Form fields:
        file              (required)  Image file (jpg/png/webp)
        model_name        (optional)  Default: RealESRGAN_x4plus
        outscale          (optional)  Default: 4
        face_enhance      (optional)  Default: false
        tile              (optional)  Default: 0
        denoise_strength  (optional)  Default: 0.5
        output_format     (optional)  png | jpg   Default: png
    """
    # --- validate upload ---
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send an image as multipart field "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    # --- parse parameters ---
    model_name = request.form.get('model_name', 'RealESRGAN_x4plus')
    if model_name not in MODEL_REGISTRY:
        return jsonify({
            'error': f'Unknown model_name "{model_name}". Valid: {VALID_MODEL_NAMES}'
        }), 400

    try:
        outscale = float(request.form.get('outscale', '4'))
    except ValueError:
        return jsonify({'error': 'outscale must be a number.'}), 400

    face_enhance = _parse_bool(request.form.get('face_enhance'), default=False)

    try:
        tile = int(request.form.get('tile', '0'))
    except ValueError:
        return jsonify({'error': 'tile must be an integer.'}), 400

    try:
        denoise_strength = float(request.form.get('denoise_strength', '0.5'))
    except ValueError:
        return jsonify({'error': 'denoise_strength must be a number.'}), 400

    output_format = request.form.get('output_format', 'png').lower()
    if output_format not in ('png', 'jpg', 'jpeg', 'webp'):
        return jsonify({'error': f'Unsupported output_format "{output_format}". Use: png, jpg, webp'}), 400

    # --- read image from upload bytes ---
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({'error': 'Could not decode the uploaded file as an image.'}), 400

    h_in, w_in = img.shape[:2]
    logger.info(
        'Upscale request: model=%s  scale=%.1f  face=%s  tile=%d  input=%dx%d  format=%s',
        model_name, outscale, face_enhance, tile, w_in, h_in, output_format,
    )

    # --- run inference (thread-safe) ---
    with _inference_lock:
        try:
            t0 = time.time()
            upsampler = get_upsampler(model_name, tile=tile, denoise_strength=denoise_strength)

            if face_enhance:
                from gfpgan import GFPGANer
                face_enhancer = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    upscale=outscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler,
                )
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True,
                )
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)

            elapsed = time.time() - t0
            h_out, w_out = output.shape[:2]
            logger.info('Upscale done: %dx%d -> %dx%d in %.2fs', w_in, h_in, w_out, h_out, elapsed)

        except RuntimeError as e:
            error_msg = str(e)
            logger.error('Inference error: %s', error_msg)
            hint = ''
            if 'out of memory' in error_msg.lower():
                hint = ' Try setting tile=256 or tile=128 to reduce GPU memory usage.'
            return jsonify({'error': f'Inference failed: {error_msg}.{hint}'}), 500
        except Exception as e:
            logger.exception('Unexpected error during inference')
            return jsonify({'error': f'Unexpected error: {e}'}), 500

    # --- encode output ---
    if output_format in ('jpg', 'jpeg'):
        ext = '.jpg'
        mime = 'image/jpeg'
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif output_format == 'webp':
        ext = '.webp'
        mime = 'image/webp'
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, 95]
    else:
        ext = '.png'
        mime = 'image/png'
        encode_params = []

    success, encoded = cv2.imencode(ext, output, encode_params)
    if not success:
        return jsonify({'error': 'Failed to encode output image.'}), 500

    # Write to a temp file and stream back
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(encoded.tobytes())
    tmp.close()

    response = send_file(
        tmp.name,
        mimetype=mime,
        as_attachment=True,
        download_name=f'upscaled{ext}',
    )

    # Clean up after response is sent
    @response.call_on_close
    def _cleanup():
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return response


@app.route('/upscale/video', methods=['POST'])
def upscale_video():
    """
    Upscale a video file.

    Form fields:
        file              (required)  Video file (mp4/avi/mkv/etc.)
        model_name        (optional)  Default: realesr-animevideov3
        outscale          (optional)  Default: 4
        face_enhance      (optional)  Default: false
        tile              (optional)  Default: 0
        denoise_strength  (optional)  Default: 0.5
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a video as multipart field "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    # --- parse parameters ---
    model_name = request.form.get('model_name', 'realesr-animevideov3')
    if model_name not in MODEL_REGISTRY:
        return jsonify({
            'error': f'Unknown model_name "{model_name}". Valid: {VALID_MODEL_NAMES}'
        }), 400

    try:
        outscale = float(request.form.get('outscale', '4'))
    except ValueError:
        return jsonify({'error': 'outscale must be a number.'}), 400

    face_enhance = _parse_bool(request.form.get('face_enhance'), default=False)

    try:
        tile = int(request.form.get('tile', '0'))
    except ValueError:
        return jsonify({'error': 'tile must be an integer.'}), 400

    try:
        denoise_strength = float(request.form.get('denoise_strength', '0.5'))
    except ValueError:
        return jsonify({'error': 'denoise_strength must be a number.'}), 400

    # --- save uploaded video to temp ---
    job_id = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(ROOT_DIR, 'tmp_video_jobs', job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    input_ext = os.path.splitext(file.filename)[1] or '.mp4'
    input_path = os.path.join(tmp_dir, f'input{input_ext}')
    file.save(input_path)

    output_path = os.path.join(tmp_dir, f'output.mp4')

    logger.info('Video upscale request: job=%s model=%s scale=%.1f', job_id, model_name, outscale)

    try:
        with _inference_lock:
            t0 = time.time()
            upsampler = get_upsampler(model_name, tile=tile, denoise_strength=denoise_strength)

            # Optional face enhancement
            face_enhancer = None
            if face_enhance:
                from gfpgan import GFPGANer
                face_enhancer = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    upscale=outscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler,
                )

            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open the uploaded video file.'}), 400

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            w_out = int(w_in * outscale)
            h_out = int(h_in * outscale)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))

            logger.info(
                'Video: %dx%d -> %dx%d, %d frames @ %.1f fps',
                w_in, h_in, w_out, h_out, total_frames, fps,
            )

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    if face_enhancer is not None:
                        _, _, enhanced = face_enhancer.enhance(
                            frame, has_aligned=False, only_center_face=False, paste_back=True,
                        )
                    else:
                        enhanced, _ = upsampler.enhance(frame, outscale=outscale)
                except RuntimeError as e:
                    logger.error('Frame %d error: %s', frame_idx, e)
                    # Write original frame resized as fallback
                    enhanced = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)

                writer.write(enhanced)
                frame_idx += 1

                if frame_idx % 10 == 0:
                    logger.info('Video progress: frame %d / %d', frame_idx, total_frames)

            cap.release()
            writer.release()
            elapsed = time.time() - t0
            logger.info('Video upscale done: %d frames in %.1fs (%.2f fps)',
                        frame_idx, elapsed, frame_idx / max(elapsed, 0.001))

    except Exception as e:
        logger.exception('Video upscale error')
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({'error': f'Video processing failed: {e}'}), 500

    # --- send back ---
    response = send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name='upscaled.mp4',
    )

    @response.call_on_close
    def _cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return response


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'error': f'File too large. Maximum upload size is {max_mb} MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found.',
        'available_endpoints': {
            'GET /health': 'Health check',
            'GET /models': 'List available models',
            'POST /upscale': 'Upscale an image',
            'POST /upscale/video': 'Upscale a video',
        },
    }), 404


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Web Service')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to. Default: 0.0.0.0')
    parser.add_argument('--port', type=int, default=8288,
                        help='Port to listen on. Default: 8288')
    parser.add_argument('-g', '--gpu-id', type=int, default=None,
                        help='GPU device to use (default=auto)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use fp32 precision. Default: fp16 (half)')
    parser.add_argument('--preload', type=str, default=None,
                        help='Preload a model at startup. E.g. --preload RealESRGAN_x4plus')
    return parser.parse_args()


def main():
    global _gpu_id, _use_fp32

    args = parse_args()
    _gpu_id = args.gpu_id
    _use_fp32 = args.fp32

    device_name = 'cpu'
    if torch.cuda.is_available():
        dev_idx = args.gpu_id or 0
        device_name = torch.cuda.get_device_name(dev_idx)

    logger.info('=' * 60)
    logger.info('Real-ESRGAN Web Service')
    logger.info('=' * 60)
    logger.info('Device     : %s', device_name)
    logger.info('FP32 mode  : %s', _use_fp32)
    logger.info('GPU ID     : %s', _gpu_id if _gpu_id is not None else 'auto')
    logger.info('Host       : %s', args.host)
    logger.info('Port       : %d', args.port)
    logger.info('=' * 60)

    # Optional: preload a model at startup
    if args.preload:
        if args.preload in MODEL_REGISTRY:
            logger.info('Preloading model: %s', args.preload)
            get_upsampler(args.preload)
        else:
            logger.warning('Unknown preload model: %s (skipping)', args.preload)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
