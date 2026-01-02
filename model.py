import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os

# ================= MEMORY SAFETY =================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = "cpu"
CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]
MODEL_PATH = os.path.join("models", "fasterrcnn_resnet18.pth")

transform = T.Compose([T.ToTensor()])

_model = None

def load_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.out_channels = 512

    model = torchvision.models.detection.FasterRCNN(
        backbone=backbone,
        num_classes=len(CLASSES),
        min_size=512
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(image_path, score_thresh=0.3, max_boxes=4):
    img = Image.open(image_path).convert("RGB")
    model = get_model()

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

    detections = []
    for box, label, score in zip(
        output["boxes"],
        output["labels"],
        output["scores"]
    ):
        if score < score_thresh:
            continue
        detections.append({
            "label": CLASSES[label],
            "score": float(score),
            "box": box.tolist()
        })

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:max_boxes]

    draw = ImageDraw.Draw(img)
    for d in detections:
        draw.rectangle(d["box"], outline="red", width=3)
        draw.text(
            (d["box"][0], d["box"][1] - 15),
            f'{d["label"]} ({d["score"]:.2f})',
            fill="white"
        )

    os.makedirs("static/outputs", exist_ok=True)
    out_path = f"static/outputs/result.png"
    img.save(out_path)

    return {
        "status": "Abnormal" if detections else "Normal",
        "detections": detections,
        "output_image": out_path
    }
