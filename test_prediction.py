
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def load_classes(class_file):
    with open(class_file, "r") as f:
        return [line.strip() for line in f.readlines()]

def predict_image(model_path, class_file, image_path):
    model = tf.keras.models.load_model(model_path)
    class_names = load_classes(class_file)

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    return class_names[predicted_index], confidence

# image to test
image_path = "test_images/test_banana.jpg"

# first model: fruit or vegetable type
item, item_confidence = predict_image(
    "models/fruit_veg_model.keras",
    "models/fruit_veg_model_classes.txt",
    image_path
)

print("\nFINAL RESULT")
print("Detected item:", item)
print("Item confidence:", round(item_confidence * 100, 2), "%")

# second model: banana ripeness only if item is banana
if item == "banana":
    ripeness, ripeness_confidence = predict_image(
        "models/banana_ripeness_model.keras",
        "models/banana_ripeness_model_classes.txt",
        image_path
    )

    print("Banana ripeness:", ripeness)
    print("Ripeness confidence:", round(ripeness_confidence * 100, 2), "%")

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