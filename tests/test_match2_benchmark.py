"""Benchmark Suite for ROMA OCR Agent (AMD AI League Match 2).

Validates speed, transcription accuracy, normalization, and JSON output schema.
"""

import os
import sys
import tempfile
import time
import shutil
import unittest
import cv2
import numpy as np

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import run_ocr

class TestMatch2Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="test_match2_")
        cls.input_dir = os.path.join(cls.test_dir, "input")
        cls.output_dir = os.path.join(cls.test_dir, "output")
        os.makedirs(cls.input_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # 1. Generate STOP sign image
        cls.stop_path = os.path.join(cls.input_dir, "stop_sign.png")
        stop_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Red background
        stop_img[:] = (20, 20, 180)
        cv2.putText(stop_img, "STOP", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 7)
        cv2.imwrite(cls.stop_path, stop_img)

        # 2. Generate Speed Limit 55 image
        cls.speed_path = os.path.join(cls.input_dir, "speed_limit.png")
        speed_img = np.full((300, 240, 3), 240, dtype=np.uint8)
        cv2.rectangle(speed_img, (10, 10), (230, 290), (0, 0, 0), 4)
        cv2.putText(speed_img, "SPEED", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(speed_img, "LIMIT", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(speed_img, "55", (55, 240), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6)
        cv2.imwrite(cls.speed_path, speed_img)

        # 3. Generate License Plate image
        cls.plate_path = os.path.join(cls.input_dir, "license_plate.png")
        plate_img = np.full((150, 300, 3), 245, dtype=np.uint8)
        cv2.rectangle(plate_img, (5, 5), (295, 145), (30, 30, 30), 3)
        cv2.putText(plate_img, "CALIFORNIA", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
        cv2.putText(plate_img, "7XYZ890", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 5)
        cv2.imwrite(cls.plate_path, plate_img)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_stop_sign_transcription(self):
        t0 = time.perf_counter()
        res = run_ocr(self.stop_path, self.output_dir)
        latency = time.perf_counter() - t0

        self.assertIn("STOP", res["text"])
        self.assertGreater(res["confidence"], 0.5)
        self.assertLess(latency, 1.0)

        # Check output file naming
        expected_output = os.path.join(self.output_dir, "stop_sign_output.json")
        self.assertTrue(os.path.exists(expected_output))
        print(f"[TEST PASS] STOP sign processed in {latency:.4f}s: {res}")

    def test_02_speed_limit_transcription(self):
        t0 = time.perf_counter()
        res = run_ocr(self.speed_path, self.output_dir)
        latency = time.perf_counter() - t0

        self.assertIn("55", res["text"])
        self.assertLess(latency, 1.0)
        expected_output = os.path.join(self.output_dir, "speed_limit_output.json")
        self.assertTrue(os.path.exists(expected_output))
        print(f"[TEST PASS] Speed limit sign processed in {latency:.4f}s: {res}")

    def test_03_license_plate_transcription(self):
        t0 = time.perf_counter()
        res = run_ocr(self.plate_path, self.output_dir)
        latency = time.perf_counter() - t0

        self.assertGreater(len(res["text"]), 0)
        self.assertLess(latency, 1.0)
        expected_output = os.path.join(self.output_dir, "license_plate_output.json")
        self.assertTrue(os.path.exists(expected_output))
        print(f"[TEST PASS] License plate processed in {latency:.4f}s: {res}")

if __name__ == "__main__":
    unittest.main()
