import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_model(dataset_path, output_model_name):
    print("\n====================================")
    print(f"Training model: {output_model_name}")
    print("Dataset path:", dataset_path)
    print("====================================\n")

    train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        len(class_names),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(MODELS_DIR / f"{output_model_name}.keras")

    with open(MODELS_DIR / f"{output_model_name}_classes.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"\nSaved model: models/{output_model_name}.keras")
    print(f"Saved classes: models/{output_model_name}_classes.txt")

    return history


# Model 1: fruit/vegetable identification
train_model(
    dataset_path="dataset/fruit_veg_balanced",
    output_model_name="fruit_veg_model"
)

# Model 2: banana ripeness classification
train_model(
    dataset_path="dataset/banana_ripeness",
    output_model_name="banana_ripeness_model"
)