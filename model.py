import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image, ImageDraw
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


# ================= MODEL LOADING =================
def load_model():
    # Backbone
    backbone = torchvision.models.resnet18(weights=None)
    backbone.out_channels = 512

    # Anchor generator (REQUIRED)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ROI Pooler (REQUIRED)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=len(CLASSES),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully")
    return model


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


# ================= PREDICTION =================
def predict(image_path, score_thresh=0.3, max_boxes=4):
    try:
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
            if score < score_thresh or label == 0:
                continue
            detections.append({
                "label": CLASSES[label],
                "score": float(score),
                "box": box.tolist()
            })

        detections = sorted(
            detections, key=lambda x: x["score"], reverse=True
        )[:max_boxes]

        # Draw boxes
        draw = ImageDraw.Draw(img)
        for d in detections:
            draw.rectangle(d["box"], outline="red", width=3)
            draw.text(
                (d["box"][0], d["box"][1] - 15),
                f"{d['label']} ({d['score']:.2f})",
                fill="white"
            )

        os.makedirs("static/outputs", exist_ok=True)
        out_path = "static/outputs/result.png"
        img.save(out_path)

        return {
            "status": "Abnormal" if detections else "Normal",
            "detections": detections,
            "output_image": out_path
        }

    except Exception as e:
        print("❌ PREDICTION ERROR:", e)
        return {
            "status": "Error",
            "reason": str(e),
            "detections": []
        }
