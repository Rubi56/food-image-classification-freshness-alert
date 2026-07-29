from pathlib import Path
import shutil

RAW_ROOT = Path("raw_datasets/banana_raw/Banana Ripeness Classification Dataset")
OUTPUT_ROOT = Path("dataset/banana_ripeness")

CLASS_MAPPING = {
    "green": "unripe",
    "ripe": "ripe",
    "overripe": "overripe",
    "rotten": "rotten"
}

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]


def is_image(path):
    return path.suffix in IMAGE_EXTENSIONS


for output_class, raw_class in CLASS_MAPPING.items():
    output_folder = OUTPUT_ROOT / output_class
    output_folder.mkdir(parents=True, exist_ok=True)

    # Clear old images first
    for old_file in output_folder.iterdir():
        if old_file.is_file():
            old_file.unlink()

    count = 0

    for split in SPLITS:
        source_folder = RAW_ROOT / split / raw_class

        if not source_folder.exists():
            print(f"Missing folder: {source_folder}")
            continue

        for img_path in source_folder.iterdir():
            if img_path.is_file() and is_image(img_path):
                new_name = f"{output_class}_{split}_{count:04d}{img_path.suffix.lower()}"
                destination = output_folder / new_name
                shutil.copy2(img_path, destination)
                count += 1

    print(f"{output_class}: copied {count} images")

print("\nBanana ripeness dataset copied successfully.")