import cv2
import numpy as np
import os
import csv

# --- Adjustable Weights for Final Score ---
WEIGHT_POSITION = 0.35
WEIGHT_ANGLE = 0.20
WEIGHT_LIGHTING = 0.25
WEIGHT_FOCUS = 0.20


##############################################
# Robust Focus Functions with Denoising
##############################################

def get_subject_roi(frame, bbox, padding=0.1):
    """
    Crop a region around the subject's bounding box with some padding.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    x1 = max(0, int(x1 - padding * box_w))
    y1 = max(0, int(y1 - padding * box_h))
    x2 = min(w, int(x2 + padding * box_w))
    y2 = min(h, int(y2 + padding * box_h))
    return frame[y1:y2, x1:x2]


def preprocess_for_focus(gray_roi):
    """
    Downsample, denoise, and mildly blur the ROI to reduce noise.
    Denoising is particularly useful for low-quality camera images.
    """
    resized = cv2.resize(gray_roi, (256, 256), interpolation=cv2.INTER_AREA)
    # Denoise to reduce sensor noise that might inflate edge measures
    denoised = cv2.fastNlMeansDenoising(resized, None, h=10, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    return blurred


def tenengrad_focus_measure(gray):
    """Compute the Tenengrad focus measure using Sobel filters."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = sobelx ** 2 + sobely ** 2
    return np.mean(grad_magnitude)


def sml_focus_measure(gray):
    """Compute the Sum of Modified Laplacian (SML) focus measure."""
    lap_x = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_y = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    shifted_x_left = np.roll(lap_x, 1, axis=1)
    shifted_x_right = np.roll(lap_x, -1, axis=1)
    mod_lap_x = np.abs(lap_x - shifted_x_left) + np.abs(lap_x - shifted_x_right)
    shifted_y_up = np.roll(lap_y, 1, axis=0)
    shifted_y_down = np.roll(lap_y, -1, axis=0)
    mod_lap_y = np.abs(lap_y - shifted_y_up) + np.abs(lap_y - shifted_y_down)
    return np.sum(mod_lap_x + mod_lap_y)


def combined_focus_measure(gray):
    """
    Combine Laplacian variance, Tenengrad, and SML focus measures.
    We use weighted contributions that you can tune empirically.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    tenengrad_val = tenengrad_focus_measure(gray)
    sml_val = sml_focus_measure(gray)
    combined = 0.3 * laplacian_var + 0.4 * tenengrad_val + 0.3 * sml_val
    return combined


def compute_focus_score(focus_measure):
    """
    Map the combined focus measure to a score between 1 and 10.
    Adjusted thresholds for low-quality cameras:
      - Focus measure below 100 → very blurry (score 1)
      - Above 800 → exceptionally sharp (score 10)
    """
    if focus_measure < 100:
        return 1
    elif focus_measure > 800:
        return 10
    else:
        ratio = (focus_measure - 100) / (800 - 100)
        scaled = 1 + ratio * (10 - 1)
        return round(scaled, 2)


##############################################
# Composition & Scene Evaluation Functions
##############################################

def calculate_position_score(focus_object, w, h, lines):
    """
    Calculate position score based on how close the subject's center is
    to ideal rule-of-thirds intersections.
    """
    if not focus_object:
        return 5

    x1, y1, x2, y2 = focus_object["bbox"]
    object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Ideal rule-of-thirds intersections (4 points)
    thirds_x = [w // 3, 2 * w // 3]
    thirds_y = [h // 3, 2 * h // 3]
    intersections = [(tx, ty) for tx in thirds_x for ty in thirds_y]

    distances = [np.sqrt((object_x - ix) ** 2 + (object_y - iy) ** 2) for (ix, iy) in intersections]
    min_distance = min(distances)
    max_possible = np.sqrt((w / 3) ** 2 + (h / 3) ** 2)
    pos_ratio = min_distance / max_possible
    position_score = round(10 - pos_ratio * 5, 2)

    # Penalize if subject occupies too much of the frame
    subject_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    if subject_area / frame_area > 0.5:
        position_score = max(3, position_score - 3)

    return position_score


def calculate_angle_score(gray):
    """
    Calculate an angle score based on the median deviation from 90°.
    Uses valid lines (between 70° and 110°) to reduce outlier effects.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 5
    angles = [abs(np.degrees(theta)) for rho, theta in lines[:, 0]]
    valid_angles = [a for a in angles if 70 <= a <= 110]
    if not valid_angles:
        return 5
    median_angle = np.median(valid_angles)
    angle_deviation = abs(median_angle - 90)
    angle_score = round(max(1, 10 - (angle_deviation / 6)), 2)
    return angle_score


def calculate_lighting_score(gray):
    """
    Calculate lighting score using both brightness and contrast.
    Ideal brightness is around 130; contrast is measured by standard deviation.
    """
    brightness = np.mean(gray)
    contrast = np.std(gray)
    brightness_score = max(1, 10 - abs(130 - brightness) / 10)
    if contrast < 30:
        contrast_score = 3
    elif contrast > 100:
        contrast_score = 4
    else:
        contrast_score = 8
    lighting_score = round((0.6 * brightness_score + 0.4 * contrast_score), 2)
    return lighting_score


def calculate_photo_score(frame, objects):
    """Evaluate the overall photo score based on composition, angle, lighting, and focus."""
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not objects:
        return {
            "Final Score": 2, "Position": 2, "Angle": 2, "Lighting": 2, "Focus": 2,
            "Feedback": ["No subject detected."], "Suggestions": ["Move subject into frame."]
        }

    angle_score = calculate_angle_score(gray)
    lighting_score = calculate_lighting_score(gray)

    focus_object = max(objects, key=lambda obj: obj["confidence"], default=None)
    if focus_object:
        roi = get_subject_roi(frame, focus_object["bbox"], padding=0.1)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        processed_roi = preprocess_for_focus(gray_roi)
        combined_measure = combined_focus_measure(processed_roi)
        focus_score = compute_focus_score(combined_measure)
    else:
        focus_score = 1

    position_score = calculate_position_score(focus_object, w, h, None)

    final_score = round(
        (position_score * WEIGHT_POSITION) +
        (angle_score * WEIGHT_ANGLE) +
        (lighting_score * WEIGHT_LIGHTING) +
        (focus_score * WEIGHT_FOCUS), 2
    )

    feedback = []
    suggestions = []

    if position_score < 5:
        feedback.append("Subject placement could improve.")
        if focus_object:
            x1, y1, x2, y2 = focus_object["bbox"]
            object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2
            if object_x < w // 3:
                suggestions.append("Move subject to the right.")
            elif object_x > 2 * w // 3:
                suggestions.append("Move subject to the left.")
            if object_y < h // 3:
                suggestions.append("Move subject downward.")
            elif object_y > 2 * h // 3:
                suggestions.append("Move subject upward.")

    if angle_score < 5:
        feedback.append("Camera tilt detected. Adjust to straighten horizon.")
        suggestions.append("Reposition camera to reduce tilt.")

    # Lighting feedback based on brightness (extracted earlier) if necessary
    brightness = np.mean(gray)
    if lighting_score < 5:
        feedback.append("Lighting is suboptimal.")
        if brightness < 100:
            suggestions.append("Increase ambient lighting or use a flash.")
        elif brightness > 180:
            suggestions.append("Reduce exposure or use shading to avoid overexposure.")
        suggestions.append("Adjust contrast if necessary.")

    if focus_score < 5:
        feedback.append("Image appears blurry.")
        suggestions.append("Stabilize camera or use a tripod.")

    return {
        "Final Score": final_score,
        "Position": position_score,
        "Angle": angle_score,
        "Lighting": lighting_score,
        "Focus": focus_score,
        "Feedback": feedback,
        "Suggestions": suggestions
    }



