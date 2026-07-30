import os
import smtplib
from email.message import EmailMessage

import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

LOW_CONFIDENCE_LIMIT = 0.50
BANANA_SAFETY_CONFIDENCE = 0.70

FOOD_CATEGORY = {
    "apple": "Fruit",
    "banana": "Fruit",
    "mango": "Fruit",
    "tomato": "Vegetable",
    "carrot": "Vegetable",
    "potato": "Vegetable"
}


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Food Image Classification & Freshness Alert System</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f5f7fa;
            margin: 40px;
        }
        .box {
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 760px;
            margin: auto;
            box-shadow: 0 0 12px #cccccc;
        }
        h2 {
            color: #2c3e50;
            text-align: center;
        }
        .section {
            margin-top: 20px;
            padding: 18px;
            background-color: #f0f3f7;
            border-radius: 10px;
        }
        .result {
            margin-top: 25px;
            padding: 18px;
            background-color: #eaf8ea;
            border-radius: 10px;
        }
        button {
            margin-top: 12px;
            padding: 10px 18px;
            cursor: pointer;
            border: none;
            border-radius: 6px;
            background-color: #2c7be5;
            color: white;
            font-size: 15px;
        }
        input {
            margin-top: 10px;
        }
        .note {
            font-size: 14px;
            color: #555555;
        }
        .message {
            font-size: 18px;
            font-weight: bold;
            color: #1b5e20;
        }
        .warning {
            font-size: 14px;
            color: #8a5a00;
        }
        .email {
            margin-top: 12px;
            font-weight: bold;
            color: #0b5394;
        }
    </style>
</head>

<body>
    <div class="box">
        <h2>AI-Powered Food Classification & Freshness Alert System</h2>

        <p>
            This Raspberry Pi system first identifies the food item.
            If the detected item is banana, it also classifies the banana ripeness stage
            and sends an email alert with a useful suggestion.
        </p>

        <div class="section">
            <h3>Option 1: Upload Image</h3>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br>
                <button type="submit">Classify Uploaded Image</button>
            </form>
        </div>

        <div class="section">
            <h3>Option 2: Camera Capture</h3>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" capture="environment" required>
                <br>
                <button type="submit">Open Camera and Classify</button>
            </form>
            <p class="note">
                Open this page on your phone browser. This option should open the phone camera
                when the phone and Raspberry Pi are on the same network.
            </p>
        </div>

        {% if result %}
        <div class="result">
            <h3>Prediction Result</h3>

            <p class="message">{{ result.message }}</p>

            <p><b>Detected Item:</b> {{ result.item }}</p>
            <p><b>Category:</b> {{ result.category }}</p>
            <p><b>Item Confidence:</b> {{ result.item_confidence }}%</p>

            {% if result.corrected %}
                <p class="warning">
                    Note: The first model had low confidence, so the banana ripeness model was also checked.
                </p>
            {% endif %}

            {% if result.ripeness %}
                <p><b>Banana Ripeness:</b> {{ result.ripeness }}</p>
                <p><b>Ripeness Confidence:</b> {{ result.ripeness_confidence }}%</p>
                <p><b>Suggestion:</b> {{ result.suggestion }}</p>
                <p><b>Recipe / Action:</b> {{ result.recipe }}</p>
                <p class="email">{{ result.email_status }}</p>
            {% else %}
                <p><b>Ripeness:</b> Not applied because the detected item is not banana.</p>
                <p class="note">Email alert is only sent for banana ripeness result.</p>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def load_classes(class_file):
    with open(class_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_tflite(model_path, class_file, image_path):
    class_names = load_classes(class_file)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    return class_names[predicted_index], confidence


def get_suggestion_and_recipe(ripeness):
    if ripeness == "green":
        suggestion = "Banana is green and not ready to eat yet. Keep it for a few more days until it ripens."
        recipe = "Condition information: store the banana at room temperature and check again later."

    elif ripeness == "ripe":
        suggestion = "Banana is ready to eat."
        recipe = "Smoothie idea: blend 1 ripe banana with milk or yogurt and a spoon of honey."

    elif ripeness == "overripe":
        suggestion = "Use it soon to avoid food waste."
        recipe = "Banana bread idea: mash 2 overripe bananas, mix with flour, egg, sugar, and oil, then bake until cooked. You can also use it for pancakes or muffins."

    elif ripeness == "rotten":
        suggestion = "Banana may not be safe to eat."
        recipe = "Check for mold, bad smell, or leaking texture. If spoiled, discard it instead of eating."

    else:
        suggestion = "No suggestion available."
        recipe = "No recipe available."

    return suggestion, recipe


def send_email_alert(ripeness, confidence, suggestion, recipe):
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not app_password or not receiver_email:
        return "Email not sent: email settings are missing."

    subject = f"Freshness Alert: Banana is {ripeness.upper()}"

    body = f"""
Hello Rubi,

Your Raspberry Pi Food Image Classification & Freshness Alert System detected a banana.

Banana ripeness: {ripeness}
Ripeness confidence: {round(float(confidence) * 100, 2)}%

Suggestion:
{suggestion}

Recipe / Action:
{recipe}

This alert was automatically generated by your AI-Powered Food Classification & Freshness Alert System.
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return "Email alert sent successfully."

    except Exception as e:
        return f"Email sending failed: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        image_path = os.path.join(UPLOAD_FOLDER, "captured_or_uploaded_image.jpg")
        file.save(image_path)

        item, item_confidence = predict_tflite(
            "models/fruit_veg_model.tflite",
            "models/fruit_veg_model_classes.txt",
            image_path
        )

        corrected = False
        ripeness = None
        ripeness_confidence = None
        suggestion = None
        recipe = None
        email_status = None

        if item == "banana":
            banana_stage, banana_stage_confidence = predict_tflite(
                "models/banana_ripeness_model.tflite",
                "models/banana_ripeness_model_classes.txt",
                image_path
            )

            ripeness = banana_stage
            ripeness_confidence = banana_stage_confidence

            suggestion, recipe = get_suggestion_and_recipe(ripeness)

            email_status = send_email_alert(
                ripeness,
                ripeness_confidence,
                suggestion,
                recipe
            )

        category = FOOD_CATEGORY.get(item, "Unknown")
        message = f"The detected item is {item.capitalize()}, which belongs to the {category.lower()} category."

        result = {
            "item": item,
            "category": category,
            "message": message,
            "item_confidence": round(float(item_confidence) * 100, 2),
            "ripeness": ripeness,
            "ripeness_confidence": round(float(ripeness_confidence) * 100, 2) if ripeness_confidence is not None else None,
            "suggestion": suggestion,
            "recipe": recipe,
            "email_status": email_status,
            "corrected": corrected
        }

    return render_template_string(HTML_PAGE, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
