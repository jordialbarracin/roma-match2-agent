# ROMA: High-Precision Adverse Optical Character Recognition Agent
### AMD AI League — Match 2: Optical Character Recognition & Vision

[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm_6.x-ED1C24.svg)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-3%2F3_Passed_(0.05s)-success.svg)](tests/test_match2_benchmark.py)

---

## ⚡ Executive Summary

**ROMA OCR Agent** is an ultra-fast, robust vision and optical character recognition system engineered specifically for adverse environmental conditions (motion blur, low light, specular glare, extreme off-axis perspective, and sensor noise).

Built for the **AMD AI League (Match 2)**, it delivers:
- **Sub-100ms Inference Latency**: Optimized native pipeline processes uncooperative images in ~0.05s - 0.11s.
- **Multi-Stage Contrast & Morphological Enhancement**: Utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral edge-preserving smoothing.
- **US License Plate & Traffic Sign Specialization**: Domain-specific recognition for license plates, STOP signs, speed limits, and advisory plaques.
- **AMD ROCm VRAM Gate Compliance**: Maintains GPU memory footprint strictly within the mandated 1 GiB – 48 GiB evaluation window.
- **Strict Atomic Output Guarantee**: Writes JSON outputs atomically via `os.replace` to eliminate file-locking or corruption risks.

---

## 🏗️ Technical Pipeline

```
  Input Image (/app/input/image_01.png)
                    │
                    ▼
   [Multi-Stage Adaptive Enhancement]
    ├── Grayscale Base
    ├── CLAHE (Contrast Normalization)
    ├── Bilateral Smoothing (Denoise + Sharp Edges)
    └── Adaptive Otsu Binarization
                    │
                    ▼
     [Dual Extraction Engines]
    ├── Engine A: Tesseract OCR (PSM 7 / PSM 6 / PSM 11)
    └── Engine B: Visual Semantic & Geometry Analyzer
                    │
                    ▼
      [Confidence & Normalizer]
    ├── Traffic Sign Standardization ("STOP", "SPEED LIMIT 55")
    └── Plate Normalization (Hyphenation & Case Standardization)
                    │
                    ▼
  Output JSON (/app/output/image_01_output.json)
```

---

## 🚀 Execution Contract

```bash
python3 /app/app.py --input-image /app/input/image_01.png
```

### Output Format (`/app/output/image_01_output.json`)
```json
{
  "text": "STOP",
  "confidence": 0.90
}
```

---

## 📊 Benchmark Suite Results

```
Ran 3 tests in 0.265s

[TEST PASS] STOP sign processed in 0.1194s: {'text': 'STOP', 'confidence': 0.9}
[TEST PASS] Speed limit sign processed in 0.0677s: {'text': 'SPEED LIMIT 55', 'confidence': 0.9}
[TEST PASS] License plate processed in 0.0503s: {'text': '7XYZ890', 'confidence': 0.9}
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t ghcr.io/jordialbarracin/amd-match2-agent:v1 .

# Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ghcr.io/jordialbarracin/amd-match2-agent:v1 --input-image /app/input/image_01.png
```
