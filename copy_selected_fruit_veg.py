from pathlib import Path
import shutil

RAW_ROOT = Path("raw_datasets/fruit_veg_raw")
OUTPUT_ROOT = Path("dataset/fruit_veg")

SELECTED_CLASSES = [
    "banana",
    "apple",
    "mango",
    "tomato",
    "carrot",
    "potato"
]

SPLITS = ["train", "test", "validation"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]


def is_image(path):
    return path.suffix in IMAGE_EXTENSIONS


for class_name in SELECTED_CLASSES:
    output_folder = OUTPUT_ROOT / class_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # remove old copied images
    for old_file in output_folder.iterdir():
        if old_file.is_file():
            old_file.unlink()

    count = 0

    for split in SPLITS:
        source_folder = RAW_ROOT / split / class_name

        if not source_folder.exists():
            print(f"Missing folder: {source_folder}")
            continue

        for img_path in source_folder.iterdir():
            if img_path.is_file() and is_image(img_path):
                new_name = f"{class_name}_{split}_{count:04d}{img_path.suffix.lower()}"
                destination = output_folder / new_name
                shutil.copy2(img_path, destination)
                count += 1

    print(f"{class_name}: copied {count} images")

print("\nSelected fruit/vegetable images copied successfully.")