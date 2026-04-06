"""Quick test script for the Real-ESRGAN web service."""
import requests
import os
import sys
import time

BASE_URL = "http://127.0.0.1:8288"

def test_health():
    print("=" * 50)
    print("TEST: GET /health")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Body: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("  PASSED")

def test_models():
    print("=" * 50)
    print("TEST: GET /models")
    resp = requests.get(f"{BASE_URL}/models")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Models: {[m['name'] for m in data['models']]}")
    assert resp.status_code == 200
    assert len(data["models"]) == 6
    print("  PASSED")

def test_upscale():
    print("=" * 50)
    print("TEST: POST /upscale")
    img_path = os.path.join(os.path.dirname(__file__), "inputs", "0014.jpg")
    if not os.path.exists(img_path):
        print(f"  SKIPPED: {img_path} not found")
        return

    print(f"  Input: {img_path} ({os.path.getsize(img_path)} bytes)")
    t0 = time.time()

    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/upscale",
            files={"file": ("0014.jpg", f, "image/jpeg")},
            data={
                "model_name": "RealESRGAN_x4plus",
                "outscale": "4",
                "output_format": "png",
            },
        )

    elapsed = time.time() - t0
    print(f"  Status: {resp.status_code}")
    print(f"  Time: {elapsed:.2f}s")

    if resp.status_code == 200:
        content_type = resp.headers.get("Content-Type", "")
        print(f"  Content-Type: {content_type}")
        print(f"  Output size: {len(resp.content)} bytes")

        out_path = os.path.join(os.path.dirname(__file__), "test_output.png")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved to: {out_path}")
        print("  PASSED")
    else:
        print(f"  Response: {resp.text}")
        print("  FAILED")

def test_404():
    print("=" * 50)
    print("TEST: GET /nonexistent (404)")
    resp = requests.get(f"{BASE_URL}/nonexistent")
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 404
    print("  PASSED")

def test_no_file():
    print("=" * 50)
    print("TEST: POST /upscale without file (400)")
    resp = requests.post(f"{BASE_URL}/upscale")
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 400
    print(f"  Error: {resp.json()['error']}")
    print("  PASSED")

if __name__ == "__main__":
    print("Real-ESRGAN Web Service Tests")
    print(f"Target: {BASE_URL}")
    print()

    test_health()
    test_models()
    test_404()
    test_no_file()
    test_upscale()

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
