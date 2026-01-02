# utils.py
from PIL import Image, ImageDraw, ImageFont
import os

def draw_boxes(image_path, detections_list, output_path, class_names):
    """
    Draw bounding boxes with labels on an image.
    detections_list: list of {"box": [...], "label": str, "score": float}
    """
    # Open image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Safe font for Linux/Render
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections_list:
        box = det.get("box")
        label = det.get("label", "Abnormal")
        score = det.get("score", 1.0)

        if not box or score < 0.5:
            continue

        x1, y1, x2, y2 = box
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw text background
        text = f"{label}: {score:.2f}"
        text_size = draw.textsize(text, font=font)
        text_y = max(y1 - text_size[1], 0)
        draw.rectangle([x1, text_y, x1 + text_size[0], text_y + text_size[1]], fill="red")
        draw.text((x1, text_y), text, fill="white", font=font)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
