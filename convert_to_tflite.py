
import tensorflow as tf
from pathlib import Path

MODELS_DIR = Path("models")


def convert_model(keras_model_path, tflite_model_path):
    print("\nConverting:", keras_model_path)

    model = tf.keras.models.load_model(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Basic optimization to make model lighter for Raspberry Pi
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print("Saved:", tflite_model_path)


# Convert fruit/vegetable model
convert_model(
    keras_model_path=MODELS_DIR / "fruit_veg_model.keras",
    tflite_model_path=MODELS_DIR / "fruit_veg_model.tflite"
)

# Convert banana ripeness model
convert_model(
    keras_model_path=MODELS_DIR / "banana_ripeness_model.keras",
    tflite_model_path=MODELS_DIR / "banana_ripeness_model.tflite"
)

print("\nBoth models converted to TensorFlow Lite successfully.")