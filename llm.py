import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_groq_response(query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.4,
        "max_tokens": 250
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return "No report generated."
        return choices[0]["message"]["content"].strip()
    except Exception as e:
        print("Groq API Error:", e)
        return "Unable to generate report."

def generate_llm_report(detections, reason=None):
    """
    Generates a concise LLM report in HTML table format with bullet points and exercises.
    - detections: list of {"label": str, "score": float, "box": [...]}
    - reason: optional string for normal cases
    Returns: HTML string
    """

    # 1️⃣ Handle normal case
    if not detections:
        return f"""
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr><th>Status</th><td>{reason or 'No lung abnormalities detected.'}</td></tr>
        </table>
        """

    # 2️⃣ Build LLM query with bullet points
    query = "You are a radiology assistant. Generate a concise chest X-ray report with findings and recommendations:\n\n"
    for d in detections:
        query += f"- {d.get('label', 'Unknown')} (confidence {d.get('score', 1.0):.2f})\n"
    query += "\nInclude brief recommendations and suggested exercises for the patient if applicable."

    llm_text = get_groq_response(query)

    # 3️⃣ Build HTML table for detections
    table_rows = ""
    for d in detections:
        table_rows += f"""
        <tr>
            <td>{d.get('label', 'Unknown')}</td>
            <td>{d.get('score', 1.0):.2f}</td>
        </tr>
        """

    # 4️⃣ Add 2 extra exercise lines at the end
    extra_exercises = [
        "Encourage deep breathing exercises to improve lung expansion.",
        "Monitor daily symptoms and report any worsening shortness of breath."
    ]

    html_report = f"""
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr>
                <th>Finding</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    <br>
    <strong>LLM Recommendations:</strong>
    <ul>
        {''.join(f'<li>{line.strip()}</li>' for line in llm_text.splitlines() if line.strip())}
        {''.join(f'<li>{line}</li>' for line in extra_exercises)}
    </ul>
    """
    return html_report
