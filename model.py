import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import torchvision.transforms as T
import os
from utils import draw_boxes  # Make sure utils.py exists

# ================= MEMORY SAFETY =================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = "cpu"
CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]

# Absolute model path for Render
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fasterrcnn_best.pth")

# Transform for images
transform = T.Compose([T.ToTensor()])

# Singleton model
_model = None

# ================= MODEL LOADING =================
def load_model():
    """Load the Faster R-CNN model with custom backbone and anchors."""
    # Backbone
    backbone = torchvision.models.resnet18(weights=None)
    backbone.out_channels = 512

    # Anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ROI Pooler
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

    # Load model weights
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully")
    return model

def get_model():
    """Singleton getter for model."""
    global _model
    if _model is None:
        _model = load_model()
    return _model

# ================= PREDICTION =================
def predict(image_path, score_thresh=0.3, max_boxes=4):
    """
    Predict lung abnormalities on a chest X-ray.

    Returns:
        dict: {
            "status": "Normal" or "Abnormal",
            "detections": list of boxes with labels and scores,
            "output_image": path to annotated image
        }
    """
    try:
        img = Image.open(image_path).convert("RGB")
        model = get_model()
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        # Collect high-confidence detections
        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score < score_thresh:
                continue
            detections.append({
                "label": CLASSES[int(label)],
                "score": float(score),
                "box": box.tolist()
            })
            if len(detections) >= max_boxes:
                break

        # Draw boxes using utils.draw_boxes
        out_path = "/tmp/outputs/result.png"  # Render-friendly
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        draw_boxes(image_path, detections, out_path, CLASSES)

        return {
            "status": "Abnormal" if detections else "Normal",
            "detections": detections,
            "output_image": out_path
        }

    except Exception as e:
        print("🔥 PREDICT ERROR 🔥", repr(e))
        raise
