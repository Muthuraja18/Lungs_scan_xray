# model.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image
import torchvision.transforms as T
import os
from utils import draw_boxes

DEVICE = "cpu"
CLASSES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]
transform = T.Compose([T.ToTensor()])
_model = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fasterrcnn_best.pth")

def load_model():
    backbone = torchvision.models.resnet18(weights=None)
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2
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
    return model

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def predict(image_path, score_thresh=0.3, max_boxes=4):
    """
    Predict lung abnormalities and draw bounding boxes.
    """
    img = Image.open(image_path).convert("RGB")
    model = get_model()
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

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

    # Save output image to static/outputs
    out_dir = "static/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    draw_boxes(image_path, detections, out_path, CLASSES)

    # Normal / Abnormal logic (unchanged)
    status = "Abnormal" if detections else "Normal"

    return {
        "status": status,
        "detections": detections,
        "output_image": out_path
    }
