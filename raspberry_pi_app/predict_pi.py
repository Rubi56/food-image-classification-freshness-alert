
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


IMG_SIZE = (224, 224)


def load_classes(class_file):
    with open(class_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_tflite(model_path, class_file, image_path):
    class_names = load_classes(class_file)

    interpreter = tflite.Interpreter(model_path=model_path)
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


image_path = "test_images/test_banana.jpg"

item, item_confidence = predict_tflite(
    "models/fruit_veg_model.tflite",
    "models/fruit_veg_model_classes.txt",
    image_path
)

print("\nFINAL RESULT")
print("Detected item:", item)
print("Item confidence:", round(float(item_confidence) * 100, 2), "%")

if item == "banana":
    ripeness, ripeness_confidence = predict_tflite(
        "models/banana_ripeness_model.tflite",
        "models/banana_ripeness_model_classes.txt",
        image_path
    )

    print("Banana ripeness:", ripeness)
    print("Ripeness confidence:", round(float(ripeness_confidence) * 100, 2), "%")

    if ripeness == "green":
        print("Suggestion: Banana is not ready yet.")
    elif ripeness == "ripe":
        print("Suggestion: Banana is ready to eat.")
    elif ripeness == "overripe":
        print("Suggestion: Use it soon for smoothie or banana bread.")
    elif ripeness == "rotten":
        print("Suggestion: Banana may not be safe to eat.")
else:
    print("Ripeness classification skipped because this is not banana.")