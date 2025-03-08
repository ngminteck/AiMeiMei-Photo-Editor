# providers/yolo_detection_provider.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8x model (load it once for reuse)
yolo_model = YOLO("yolov8x.pt")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def are_close(box1, box2, distance_threshold=30000, iou_threshold=0.2):
    # Compute center points
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
    center2 = ((bx1 + bx2) // 2, (by1 + by2) // 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    iou_val = compute_iou(box1, box2)
    return (distance < distance_threshold) or (iou_val > iou_threshold)

def detect_objects(frame):
    """
    Run YOLO detection on the input frame and return a list of detected objects.
    Each detection is a dict with keys:
      - 'bbox': (x1, y1, x2, y2)
      - 'confidence': detection confidence
      - 'label': object label
    """
    results = yolo_model(frame)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
            label = yolo_model.names[class_id] if class_id is not None else "object"
            detected_objects.append({
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "label": label
            })
    return detected_objects

def group_objects(objects):
    """
    Group nearby objects using center distance and IoU.
    Returns a list of groups (clusters) with a merged bounding box.
    """
    grouped_clusters = []
    used = set()
    for i, obj in enumerate(objects):
        if i in used:
            continue
        group = [obj]
        used.add(i)
        for j, other_obj in enumerate(objects):
            if j not in used and are_close(obj["bbox"], other_obj["bbox"]):
                group.append(other_obj)
                used.add(j)
        # Merge bounding boxes
        x1 = min(o["bbox"][0] for o in group)
        y1 = min(o["bbox"][1] for o in group)
        x2 = max(o["bbox"][2] for o in group)
        y2 = max(o["bbox"][3] for o in group)
        grouped_clusters.append({
            "confidence": max(o["confidence"] for o in group),
            "bbox": (x1, y1, x2, y2),
            "members": group
        })
    return grouped_clusters

def select_focus_object(grouped_clusters, frame_shape):
    """
    From grouped clusters, select the focus object based on area, confidence, and distance from the frame center.
    """
    if not grouped_clusters:
        return None
    frame_height, frame_width = frame_shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)
    best_score = -1
    best_group = None
    for group in grouped_clusters:
        x1, y1, x2, y2 = group["bbox"]
        area = (x2 - x1) * (y2 - y1)
        group_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = np.sqrt((group_center[0] - frame_center[0]) ** 2 + (group_center[1] - frame_center[1]) ** 2)
        confidence = group["confidence"]
        score = area * confidence / (distance + 1)
        if score > best_score:
            best_score = score
            best_group = group
    return best_group

def draw_bounding_box(frame, group):
    """
    Draws individual (blue) boxes for members and a grouped (red) bounding box.
    """
    if group is None:
        return frame

    blue = (255, 0, 0)  # For individual boxes
    red = (0, 0, 255)   # For the grouped focus box
    scale_factor = 0.8  # Scale down for individual boxes

    for member in group.get("members", []):
        bx1, by1, bx2, by2 = member["bbox"]
        width = bx2 - bx1
        height = by2 - by1
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        center_x = bx1 + width // 2
        center_y = by1 + height // 2
        new_bx1 = center_x - new_width // 2
        new_by1 = center_y - new_height // 2
        new_bx2 = new_bx1 + new_width
        new_by2 = new_by1 + new_height

        cv2.rectangle(frame, (new_bx1, new_by1), (new_bx2, new_by2), blue, 2)
        label = member.get("label", "object")
        confidence = member.get("confidence", 0)
        debug_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, debug_text, (new_bx1, new_by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1, cv2.LINE_AA)

    # Draw the overall grouped bounding box in red
    x1, y1, x2, y2 = group["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 3)
    return frame

def detect_main_object(frame):
    """
    Runs detection on the frame, groups detections, selects the focus group, and draws bounding boxes.
    Returns the modified frame and the focus group.
    """
    objects = detect_objects(frame)
    if not objects:
        return frame, None
    grouped_clusters = group_objects(objects)
    focus_group = select_focus_object(grouped_clusters, frame.shape)
    frame = draw_bounding_box(frame, focus_group)
    return frame, focus_group
