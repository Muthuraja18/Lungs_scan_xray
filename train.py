import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import os

# ===== Settings =====
DEVICE = "cpu"
BATCH_SIZE = 8
EPOCHS = 5
DATA_DIR = "dataset"

# ===== Transforms =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== Dataset =====
train_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ===== Model =====
model = torchvision.models.mobilenet_v3_small(
    weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

# 🔁 Change final layer (2 classes)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.to(DEVICE)

# ===== Training Setup =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ===== Train =====
print("🚀 Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ===== Save Model =====
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mobilenet_xray.pth")

print("✅ Model saved as models/mobilenet_xray.pth")
