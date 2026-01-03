import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os
import numpy as np

DEVICE = "cpu"

# Classes
CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]

# Path to your trained model

MODEL_PATH = os.path.join("models", "mobilenet_xray.pth")

# ---------------------------
# Load model
# ---------------------------
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
transform = T.Compose([T.ToTensor()])

def is_ct_scan(img):
    """
    Simple check to reject CT slices based on shape and contrast.
    Returns True if image is likely a CT slice, False otherwise.
    """
    arr = np.array(img.convert("L"))
    h, w = arr.shape
    aspect_ratio = w / h
    contrast = arr.std()

    # CT slices are usually square and have high contrast
    if 0.9 < aspect_ratio < 1.1 and contrast > 70:
        return True
    return False

# ---------------------------
# Lung ROI helpers
# ---------------------------
def lung_roi(img):
    w, h = img.size
    return int(0.15*w), int(0.20*h), int(0.85*w), int(0.85*h)

def inside_lung(box, roi):
    x1, y1, x2, y2 = box
    lx1, ly1, lx2, ly2 = roi
    cx, cy = (x1+x2)/2, (y1+y2)/2
    return lx1 <= cx <= lx2 and ly1 <= cy <= ly2

def valid_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    if bw < 30 or bh < 30 or bw > 0.6*img_w or bh > 0.6*img_h:
        return False
    return True


# ---------------------------
# Prediction
def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print("IMAGE LOAD ERROR:", e)
        return None

# ---------------------------
def predict(image_path, score_thresh=0.03, max_boxes=4):
    img = safe_open_image(image_path)
    if img is None:
        return {"status": "Error", "reason": "Invalid image", "detections": []}

    # Reject CT scans
    if is_ct_scan(img):
        return {
            "status": "Error",
            "reason": "CT image detected. Please upload chest X-ray only.",
            "detections": []
        }

    img_tensor = transform(img).unsqueeze(0)

    try:
        with torch.no_grad():
            output = model(img_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        roi = lung_roi(img)
        detections = []

        # Filter boxes
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue
            cls = CLASSES[label]
            if cls == "__background__":
                continue
            box = box.tolist()
            if not inside_lung(box, roi):
                continue
            if not valid_box(box, img.size[0], img.size[1]):
                continue
            detections.append({
                "label": cls,
                "score": float(score),
                "box": box
            })

        # Limit to max_boxes
        if len(detections) > max_boxes:
            # Keep top max_boxes by score
            detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:max_boxes]

        # -----------------------------
        # Normal vs Abnormal logic (UNCHANGED)
        # -----------------------------
        if len(detections) == 0:
            status = "Normal"
            reason = "No focal lung abnormality detected."
        else:
            status = "Abnormal"
            reason = "Localized lung abnormality detected."

        # Draw boxes
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{det['label']} ({det['score']:.2f})"
            try:
                text_width, text_height = font.getsize(text)
            except AttributeError:
                bbox = draw.textbbox((0,0), text, font=font)
                text_width, text_height = bbox[2]-bbox[0], bbox[3]-bbox[1]

            draw.rectangle([x1, y1-text_height, x1+text_width, y1], fill="red")
            draw.text((x1, y1-text_height), text, fill="white", font=font)

        # Save annotated image
        out_path = image_path.replace(".jpg", "_pred.jpg").replace(".png", "_pred.png")
        img.save(out_path)

        return {
            "status": status,
            "reason": reason,
            "detections": detections,
            "output_image": out_path
        }

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return {
            "status": "Error",
            "reason": "Processing failed",
            "detections": []
        }


