import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os
import numpy as np

# ---------------------------
# Memory safety (CRITICAL)
# ---------------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = "cpu"
CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]
MODEL_PATH = os.path.join("models", "fasterrcnn_resnet18.pth")

transform = T.Compose([T.ToTensor()])

# ---------------------------
# Lazy-loaded model (singleton)
# ---------------------------
_model = None

def load_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.out_channels = 512

    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=len(CLASSES),
        min_size=512
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

# ---------------------------
# Prediction
# ---------------------------
def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def predict(image_path, score_thresh=0.03, max_boxes=4):
    img = safe_open_image(image_path)
    if img is None:
        return {"status": "Error", "reason": "Invalid image", "detections": []}

    model = get_model()  # ✅ load only once
    img_tensor = transform(img).unsqueeze(0)

    try:
        with torch.no_grad():
            output = model(img_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh or CLASSES[label] == "__background__":
                continue
            detections.append({
                "label": CLASSES[label],
                "score": float(score),
                "box": box.tolist()
            })

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:max_boxes]

        # Draw boxes
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 15),
                      f"{det['label']} ({det['score']:.2f})",
                      fill="white",
                      font=font)

        out_dir = os.path.join("static", "outputs")
        os.makedirs(out_dir, exist_ok=True)

        name, ext = os.path.splitext(os.path.basename(image_path))
        out_path = os.path.join(out_dir, f"{name}_pred{ext}")
        img.save(out_path)

        return {
            "status": "Abnormal" if detections else "Normal",
            "reason": "Localized lung abnormality detected." if detections else "No focal lung abnormality detected.",
            "detections": detections,
            "output_image": out_path
        }

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return {"status": "Error", "reason": "Processing failed", "detections": []}
