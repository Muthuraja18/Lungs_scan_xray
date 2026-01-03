import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import os

# ========= MEMORY SAFETY =========
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = "cpu"
CLASS_NAMES = ["Normal", "Abnormal"]

# ---------- Transforms ----------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Load pretrained MobileNet ----------
print("🔵 Loading pretrained MobileNetV3...")

model = torchvision.models.mobilenet_v3_small(
    weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

model.eval()

print("🟢 Model ready (pretrained)")


# ---------- Prediction ----------
def predict(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512))

        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        # Simple abnormality heuristic
        abnormal_score = float(torch.mean(probs[1:]))  # non-background
        status = "Abnormal" if abnormal_score > 0.35 else "Normal"

        return {
            "status": status,
            "confidence": round(abnormal_score, 3),
            "detections": [],
            "output_image": image_path
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        return {
            "status": "Error",
            "confidence": 0.0,
            "detections": [],
            "output_image": None
        }
