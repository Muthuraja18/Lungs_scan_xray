from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image_path, detections_list, output_path, class_names):
    """
    Draws bounding boxes on the image.
    detections_list: list of {"box": [...], "label": str, "score": float}
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections_list:
        box = det.get("box")
        label = det.get("label", "Abnormal")
        score = det.get("score", 1.0)

        if not box:
            continue

        if score < 0.5:  # Skip low-confidence boxes
            continue

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 20), f"{label}: {score:.2f}", fill="red", font=font)

    img.save(output_path)
