"""
Image Validation Module for Melanoma Detection System
Validates if uploaded images are suitable for medical analysis
"""

import numpy as np
from PIL import Image
import cv2

class ImageValidator:
    """Validates if an image is suitable for melanoma detection"""
    
    def __init__(self):
        self.min_skin_tone_percentage = 0.25  # At least 25% skin-like colors (BALANCED)
        self.max_uniform_percentage = 0.85    # Not more than 85% uniform color
        self.min_complexity = 0.04             # Minimum texture complexity
        self.min_required_checks = 4           # Need to pass 4 out of 6 checks (BALANCED)
        
    def validate_image(self, image_path):
        """
        Validate if image is suitable for melanoma detection
        
        Returns:
            dict: {
                'is_valid': bool,
                'confidence': float,
                'reason': str,
                'checks': dict
            }
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Run validation checks
            checks = {}
            
            # Check 1: Skin tone detection
            skin_check = self._check_skin_tones(img_array)
            checks['has_skin_tones'] = skin_check
            
            # Check 2: Image complexity (not too simple/uniform)
            complexity_check = self._check_complexity(img_array)
            checks['has_complexity'] = complexity_check
            
            # Check 3: Not text/document
            text_check = self._check_not_text(img_array)
            checks['not_text'] = text_check
            
            # Check 4: Appropriate color distribution
            color_check = self._check_color_distribution(img_array)
            checks['color_distribution'] = color_check
            
            # Check 5: Edge density (medical images have moderate edges)
            edge_check = self._check_edge_density(img_array)
            checks['edge_density'] = edge_check
            
            # Check 6: Not sharp manufactured objects (pens, tools, etc.)
            object_check = self._check_not_sharp_objects(img_array)
            checks['not_sharp_objects'] = object_check
            
            # Calculate overall validity
            passed_checks = sum([
                checks['has_skin_tones']['passed'],
                checks['has_complexity']['passed'],
                checks['not_text']['passed'],
                checks['color_distribution']['passed'],
                checks['edge_density']['passed'],
                checks['not_sharp_objects']['passed']
            ])
            
            total_checks = 6
            confidence = (passed_checks / total_checks) * 100
            
            # Determine if valid - needs to pass 4/6 checks (more lenient for medical images)
            is_valid = (passed_checks >= self.min_required_checks)
            
            # Determine reason for rejection
            if not is_valid:
                if not checks['has_skin_tones']['passed']:
                    reason = "❌ Image does not contain skin-like colors. Please upload a photo of actual skin/lesion."
                elif not checks['not_sharp_objects']['passed']:
                    reason = "❌ Image appears to be a manufactured object (pen, tool, etc.), not a skin lesion."
                elif not checks['has_complexity']['passed']:
                    reason = "❌ Image is too uniform or simple. Please upload a clear photo of a skin lesion."
                elif not checks['not_text']['passed']:
                    reason = "❌ Image appears to be a document or text. Please upload a photo of a skin lesion."
                elif not checks['color_distribution']['passed']:
                    reason = "❌ Image has unusual colors for medical analysis. Please upload a skin lesion photo."
                else:
                    reason = "❌ Image does not appear to be a valid skin lesion photo. Please try another image."
            else:
                reason = "✅ Image appears to be a valid medical image for analysis."
            
            return {
                'is_valid': is_valid,
                'confidence': round(confidence, 2),
                'reason': reason,
                'checks': checks,
                'passed_checks': f"{passed_checks}/{total_checks}",
                'detailed_results': {
                    'skin_tones': checks['has_skin_tones']['message'],
                    'complexity': checks['has_complexity']['message'],
                    'not_text': checks['not_text']['message'],
                    'color_dist': checks['color_distribution']['message'],
                    'edges': checks['edge_density']['message'],
                    'not_objects': checks['not_sharp_objects']['message']
                }
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'confidence': 0,
                'reason': f"Error validating image: {str(e)}",
                'checks': {},
                'passed_checks': "0/5"
            }
    
    def _check_skin_tones(self, img_array):
        """Check if image contains skin-like colors"""
        try:
            # Convert to HSV for better skin detection
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Multiple skin tone ranges in HSV (covers various skin types)
            # Range 1: Light to medium skin (peach/beige tones)
            lower_skin1 = np.array([0, 25, 80], dtype=np.uint8)
            upper_skin1 = np.array([20, 170, 255], dtype=np.uint8)
            
            # Range 2: Medium skin tones
            lower_skin2 = np.array([0, 30, 60], dtype=np.uint8)
            upper_skin2 = np.array([25, 130, 220], dtype=np.uint8)
            
            # Range 3: Darker skin tones
            lower_skin3 = np.array([0, 15, 40], dtype=np.uint8)
            upper_skin3 = np.array([28, 100, 180], dtype=np.uint8)
            
            # Create masks
            mask1 = cv2.inRange(img_hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(img_hsv, lower_skin2, upper_skin2)
            mask3 = cv2.inRange(img_hsv, lower_skin3, upper_skin3)
            
            # Combine all masks
            skin_mask = cv2.bitwise_or(mask1, mask2)
            skin_mask = cv2.bitwise_or(skin_mask, mask3)
            
            # Calculate percentage of skin-like pixels
            skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
            
            # Also check if skin tones are in realistic RGB ranges
            # Skin typically has more red than blue
            r_channel = img_array[:, :, 0]
            b_channel = img_array[:, :, 2]
            r_greater_than_b = np.mean(r_channel > b_channel)
            
            # Skin images should have reasonable red dominance
            has_skin_rgb_pattern = r_greater_than_b > 0.4
            
            passed = (skin_percentage >= self.min_skin_tone_percentage) and has_skin_rgb_pattern
            
            return {
                'passed': passed,
                'value': round(skin_percentage * 100, 2),
                'threshold': self.min_skin_tone_percentage * 100,
                'message': f"Skin-like colors: {skin_percentage*100:.1f}% (need ≥{self.min_skin_tone_percentage*100:.1f}%), R>B: {r_greater_than_b*100:.1f}%"
            }
        except:
            return {'passed': False, 'value': 0, 'threshold': 25, 'message': 'Skin tone check failed'}
    
    def _check_complexity(self, img_array):
        """Check if image has sufficient complexity (not too uniform)"""
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate standard deviation (measure of variation)
            std_dev = np.std(gray) / 255.0  # Normalize to 0-1
            
            # Check color uniformity
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            color_diversity = unique_colors / total_pixels
            
            # Image should not be too uniform
            uniform_percentage = 1 - color_diversity
            
            passed = (std_dev >= self.min_complexity) and (uniform_percentage <= self.max_uniform_percentage)
            
            return {
                'passed': passed,
                'value': round(std_dev, 3),
                'threshold': self.min_complexity,
                'message': f"Complexity: {std_dev:.3f} (need ≥{self.min_complexity}), Uniformity: {uniform_percentage*100:.1f}%"
            }
        except:
            return {'passed': True, 'value': 0.5, 'threshold': 0.05, 'message': 'Complexity check skipped'}
    
    def _check_not_text(self, img_array):
        """Check if image is not primarily text/document"""
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Count transitions (text has many small objects with sharp transitions)
            horizontal_transitions = np.sum(np.abs(np.diff(binary, axis=1)) > 100)
            vertical_transitions = np.sum(np.abs(np.diff(binary, axis=0)) > 100)
            total_transitions = horizontal_transitions + vertical_transitions
            
            # Normalize by image size
            transition_density = total_transitions / (gray.shape[0] * gray.shape[1])
            
            # Text/documents typically have high transition density
            # Medical images have moderate density
            passed = transition_density < 0.5  # Less than 50% transitions
            
            return {
                'passed': passed,
                'value': round(transition_density, 3),
                'threshold': 0.5,
                'message': f"Transition density: {transition_density:.3f} (text usually >0.5)"
            }
        except:
            return {'passed': True, 'value': 0.2, 'threshold': 0.5, 'message': 'Text check skipped'}
    
    def _check_color_distribution(self, img_array):
        """Check if color distribution is appropriate for medical images"""
        try:
            # Calculate color channel statistics
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # Check if colors are too extreme or too uniform
            # Medical images typically have balanced colors with slight variations
            mean_values = [r_mean, g_mean, b_mean]
            color_balance = max(mean_values) - min(mean_values)
            
            # Check for unnatural colors (e.g., pure digital graphics)
            is_too_saturated = any(m > 240 or m < 15 for m in mean_values)
            has_reasonable_balance = color_balance < 150  # Not too imbalanced
            
            passed = has_reasonable_balance and not is_too_saturated
            
            return {
                'passed': passed,
                'value': round(color_balance, 2),
                'threshold': 150,
                'message': f"Color balance: {color_balance:.1f} (R:{r_mean:.0f}, G:{g_mean:.0f}, B:{b_mean:.0f})"
            }
        except:
            return {'passed': True, 'value': 50, 'threshold': 150, 'message': 'Color check skipped'}
    
    def _check_edge_density(self, img_array):
        """Check edge density - medical images have moderate edges"""
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_percentage = np.sum(edges > 0) / edges.size
            
            # Medical images: not too few edges (blank), not too many (text/noise)
            passed = 0.02 <= edge_percentage <= 0.30
            
            return {
                'passed': passed,
                'value': round(edge_percentage * 100, 2),
                'threshold': "2-30%",
                'message': f"Edge density: {edge_percentage*100:.1f}% (medical images: 2-30%)"
            }
        except:
            return {'passed': True, 'value': 10, 'threshold': "2-30%", 'message': 'Edge check skipped'}
    
    def _check_not_sharp_objects(self, img_array):
        """Enhanced check to detect and reject non-medical objects"""
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 1. Detect straight lines (shoes, furniture, pens, tools)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                    minLineLength=40, maxLineGap=10)
            
            long_lines = 0
            parallel_lines = 0
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 80:
                        long_lines += 1
                    
                    # Check for parallel lines (characteristic of manufactured objects)
                    if i < len(lines) - 1:
                        angle1 = np.arctan2(y2-y1, x2-x1)
                        for j in range(i+1, min(i+5, len(lines))):
                            x3, y3, x4, y4 = lines[j][0]
                            angle2 = np.arctan2(y4-y3, x4-x3)
                            if abs(angle1 - angle2) < 0.1:  # Nearly parallel
                                parallel_lines += 1
            
            # 2. Check for geometric shapes (circles, rectangles - non-organic)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            geometric_shapes = 0
            for contour in contours:
                if len(contour) > 5:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        # Check circularity (perfect circles = manufactured)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.9:  # Very circular = manufactured
                                geometric_shapes += 1
            
            # 3. Check surface smoothness (plastic, metal, leather)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            smoothness = np.std(gray - blur)
            is_very_smooth = smoothness < 4  # Skin has texture, objects are smooth
            
            # 4. Check color uniformity (shoes, furniture have uniform colors)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            is_uniform_color = (h_std < 15 and s_std < 30)
            
            # 5. Check for sharp color boundaries (objects have hard edges)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            color_gradient = np.std(np.gradient(l_channel))
            has_sharp_boundaries = color_gradient > 45
            
            # 6. Check texture pattern (organic vs manufactured)
            # Skin lesions have irregular texture, objects have regular patterns
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            is_too_regular = texture_variance > 1000  # High variance = sharp edges/manufactured
            
            # REJECTION CRITERIA - More balanced approach
            rejection_reasons = []
            
            # Strong indicators of manufactured objects
            if long_lines > 4:
                rejection_reasons.append(f"Too many straight lines ({long_lines})")
            if parallel_lines > 2:
                rejection_reasons.append(f"Multiple parallel lines ({parallel_lines})")
            if geometric_shapes > 1:
                rejection_reasons.append("Multiple perfect shapes detected")
            if is_very_smooth and is_uniform_color and long_lines > 1:
                rejection_reasons.append("Smooth uniform surface with edges (manufactured)")
            if is_uniform_color and has_sharp_boundaries and long_lines > 2:
                rejection_reasons.append("Hard edges with uniform color (object)")
            
            is_manufactured = len(rejection_reasons) > 0
            passed = not is_manufactured
            
            message = f"Lines: {long_lines}, Smooth: {smoothness:.1f}, Shapes: {geometric_shapes}"
            if rejection_reasons:
                message += f" | REJECTED: {'; '.join(rejection_reasons)}"
            
            return {
                'passed': passed,
                'value': long_lines,
                'threshold': "No manufactured objects",
                'message': message
            }
        except Exception as e:
            return {'passed': True, 'value': 0, 'threshold': "N/A", 'message': f'Object check skipped: {str(e)}'}


# Create global validator instance
validator = ImageValidator()


def validate_uploaded_image(image_path):
    """
    Convenience function to validate an uploaded image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Validation results
    """
    return validator.validate_image(image_path)
