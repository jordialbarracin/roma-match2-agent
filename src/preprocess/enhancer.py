"""Adaptive Image Enhancement Pipeline for Adverse Optical Conditions.

Optimized for:
- Glare, specular reflections, and low-light conditions.
- Sensor noise and motion blur.
- High-contrast edge preservation for license plates and road signs.
"""

import cv2
import numpy as np
from typing import Tuple, List

class ImageEnhancer:
    @staticmethod
    def enhance_for_ocr(image_path: str) -> List[np.ndarray]:
        """Loads and generates multiple enhanced variations of the image for robust OCR."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")

        variations = []

        # 1. Grayscale base
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variations.append(gray)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for glare/shadows
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_clahe = clahe.apply(gray)
        variations.append(enhanced_clahe)

        # 3. Bilateral filter (denoise while preserving sharp character boundaries)
        denoised = cv2.bilateralFilter(enhanced_clahe, d=9, sigmaColor=75, sigmaSpace=75)
        variations.append(denoised)

        # 4. Otsu adaptive binarization
        _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variations.append(binarized)

        # 5. Inverted binary (for white-on-dark signs like STOP or advisory signs)
        inverted = cv2.bitwise_not(binarized)
        variations.append(inverted)

        return variations
