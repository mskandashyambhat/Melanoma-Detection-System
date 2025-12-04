"""
Feature Extraction Module for Melanoma Detection
Extracts ABCD (Asymmetry, Border, Color, Diameter) and other dermoscopic features
"""

import numpy as np
import cv2
from PIL import Image


class FeatureExtractor:
    """Extract dermoscopic features from skin lesion images"""
    
    def __init__(self):
        pass
    
    def extract_features(self, image_path):
        """
        Extract comprehensive features from image
        
        Returns:
            dict: Dictionary containing all extracted features
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Normalize
            img_normalized = img_array.astype(np.float32) / 255.0
            
            # Extract all features
            features = {}
            
            # 1. ABCD Features
            features['abcd'] = self._extract_abcd(img_normalized)
            
            # 2. Color features
            features['color'] = self._extract_color_features(img_normalized)
            
            # 3. Texture features
            features['texture'] = self._extract_texture_features(img_normalized)
            
            # 4. Shape features
            features['shape'] = self._extract_shape_features(img_normalized)
            
            # 5. Lesion characteristics
            features['lesion'] = self._extract_lesion_characteristics(img_normalized)
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return {}
    
    def _extract_abcd(self, img_normalized):
        """Extract ABCD dermoscopic criteria"""
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Asymmetry: Compare left-right and top-bottom symmetry
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        right_flipped = np.fliplr(right_half)
        
        # Pad to match size
        if right_flipped.shape[1] < left_half.shape[1]:
            left_half = left_half[:, :right_flipped.shape[1]]
        elif right_flipped.shape[1] > left_half.shape[1]:
            right_flipped = right_flipped[:, :left_half.shape[1]]
        
        asymmetry_lr = np.mean(np.abs(left_half - right_flipped)) / 255.0
        
        # Top-bottom asymmetry
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        bottom_flipped = np.flipud(bottom_half)
        
        if bottom_flipped.shape[0] < top_half.shape[0]:
            top_half = top_half[:bottom_flipped.shape[0], :]
        elif bottom_flipped.shape[0] > top_half.shape[0]:
            bottom_flipped = bottom_flipped[:top_half.shape[0], :]
        
        asymmetry_tb = np.mean(np.abs(top_half - bottom_flipped)) / 255.0
        
        asymmetry = (asymmetry_lr + asymmetry_tb) / 2
        
        # Border irregularity: Calculate gradient at border
        edges = cv2.Canny(gray, 50, 150)
        border_irregularity = np.std(np.sum(edges, axis=0)) + np.std(np.sum(edges, axis=1))
        border_irregularity = min(1.0, border_irregularity / 100)
        
        # Color: Number of distinct colors in lesion
        # Threshold to get lesion region
        _, binary = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        
        # Extract lesion region
        lesion_img = img_uint8.copy()
        lesion_img[binary == 0] = [0, 0, 0]
        
        # Count dominant colors in lesion (quantize to 16 colors)
        lesion_pixels = img_normalized[binary > 0]
        if len(lesion_pixels) > 0:
            # Reshape for clustering
            pixel_data = lesion_pixels.reshape(-1, 3)
            # Simple color count by rounding to nearest 0.2 intervals
            unique_colors = len(np.unique(np.round(pixel_data * 5), axis=0))
            color_count = min(6, unique_colors / 50)  # Normalize to 0-6 scale
        else:
            color_count = 0
        
        # Diameter: Estimate from lesion size
        lesion_pixels_count = np.sum(binary > 0)
        diameter = np.sqrt(lesion_pixels_count / np.pi) * 2
        diameter_normalized = min(1.0, diameter / 200)  # Normalize assuming max ~200px
        
        return {
            'asymmetry': float(asymmetry),
            'border_irregularity': float(border_irregularity),
            'color_variety': float(color_count),
            'diameter': float(diameter_normalized)
        }
    
    def _extract_color_features(self, img_normalized):
        """Extract color-based features"""
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        # Get average colors
        r_mean = np.mean(img_normalized[:, :, 0])
        g_mean = np.mean(img_normalized[:, :, 1])
        b_mean = np.mean(img_normalized[:, :, 2])
        
        # Color saturation
        rgb_max = np.max(img_normalized, axis=2)
        rgb_min = np.min(img_normalized, axis=2)
        saturation = np.mean((rgb_max - rgb_min) / (rgb_max + 1e-5))
        
        # Convert to HSV for hue analysis
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0].astype(np.float32) / 180.0
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        
        hue_variance = np.std(h_channel)
        value_mean = np.mean(v_channel)
        
        return {
            'red_mean': float(r_mean),
            'green_mean': float(g_mean),
            'blue_mean': float(b_mean),
            'saturation': float(saturation),
            'hue_variance': float(hue_variance),
            'brightness': float(value_mean)
        }
    
    def _extract_texture_features(self, img_normalized):
        """Extract texture-based features"""
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Contrast
        contrast = np.std(gray) / 255.0
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = np.std(laplacian) / 255.0
        
        # Local binary pattern (simple approximation)
        # Calculate local variations
        local_var = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        roughness = np.mean(np.abs(local_var)) / 255.0
        
        return {
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity),
            'roughness': float(roughness)
        }
    
    def _extract_shape_features(self, img_normalized):
        """Extract shape-based features"""
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            
            # Area
            area = cv2.contourArea(cnt)
            
            # Perimeter
            perimeter = cv2.arcLength(cnt, True)
            
            # Circularity (4π*Area / Perimeter²)
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
            
            # Compactness
            compactness = 1.0 - circularity
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-5)
            
            # Extent
            x, y, w, h = cv2.boundingRect(cnt)
            extent = area / (w * h + 1e-5)
            
        else:
            area = circularity = compactness = solidity = extent = 0
        
        return {
            'circularity': float(circularity),
            'compactness': float(compactness),
            'solidity': float(solidity),
            'extent': float(extent),
            'area': float(area)
        }
    
    def _extract_lesion_characteristics(self, img_normalized):
        """Extract overall lesion characteristics"""
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Mean intensity
        mean_intensity = np.mean(gray) / 255.0
        
        # Intensity variance
        intensity_variance = np.std(gray) / 255.0
        
        # Number of peaks in histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)
        
        # Count local maxima
        peaks = 0
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i, 0] > hist_smooth[i-1, 0] and hist_smooth[i, 0] > hist_smooth[i+1, 0]:
                peaks += 1
        
        return {
            'mean_intensity': float(mean_intensity),
            'intensity_variance': float(intensity_variance),
            'histogram_peaks': float(peaks),
            'overall_darkness': float(1 - mean_intensity)
        }


# Create global extractor
feature_extractor = FeatureExtractor()


def extract_image_features(image_path):
    """Convenience function to extract features"""
    return feature_extractor.extract_features(image_path)
