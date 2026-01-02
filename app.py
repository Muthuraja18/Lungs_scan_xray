from flask import Flask, request, jsonify, render_template
import os
from model import predict
from utils import draw_boxes
from llm import generate_llm_report
from PIL import Image

app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLASS_NAMES = ["__background__", "opacity", "nodule", "consolidation", "effusion"]

def normalize_detections(detections, class_names):
    normalized = []
    for d in detections:
        box = d.get("box") or (d.get("boxes")[0] if d.get("boxes") else None)
        label_idx = d.get("label") or d.get("label_id") or (d.get("labels")[0] if d.get("labels") else 0)
        label_name = class_names[label_idx] if isinstance(label_idx, int) and label_idx < len(class_names) else str(label_idx)
        score = d.get("score") or (d.get("scores")[0] if d.get("scores") else 1.0)
        if box:
            normalized.append({"label": label_name, "score": score, "box": box})
    return normalized

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"})

        img_filename = file.filename
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)
        out_path = os.path.join(OUTPUT_FOLDER, img_filename)
        file.save(img_path)

        result = predict(img_path)
        normalized_detections = normalize_detections(result.get("detections", []), CLASS_NAMES)

        if normalized_detections:
            draw_boxes(img_path, normalized_detections, out_path, CLASS_NAMES)
            output_image = f"/static/outputs/{img_filename}"
            abnormal = True
        else:
            output_image = f"/static/uploads/{img_filename}"
            abnormal = False

        llm_report = generate_llm_report(
            normalized_detections,
            reason=result.get("reason") if abnormal else "No lung abnormalities detected."
        )

        return jsonify({
            "output": output_image,
            "abnormal": abnormal,
            "report": llm_report,
            "detections": normalized_detections
        })

    except Exception as e:
        import traceback
        print("UPLOAD ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
