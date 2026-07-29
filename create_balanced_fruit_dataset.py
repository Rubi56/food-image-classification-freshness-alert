import shutil
from pathlib import Path
import random

random.seed(42)

project_dir = Path("/home/rubi/Documents/freshness_alert_project")

old_fruit_dir = project_dir / "dataset" / "fruit_veg"
banana_ripeness_dir = project_dir / "dataset" / "banana_ripeness"
balanced_dir = project_dir / "dataset" / "fruit_veg_balanced"

classes = ["apple", "mango", "carrot", "potato", "tomato"]

if balanced_dir.exists():
    shutil.rmtree(balanced_dir)

balanced_dir.mkdir(parents=True, exist_ok=True)

def copy_images(source_dir, dest_dir, prefix, max_count):
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(source_dir.glob(ext))

    random.shuffle(images)
    selected = images[:max_count]

    for i, img in enumerate(selected):
        shutil.copy2(img, dest_dir / f"{prefix}_{i}_{img.name}")

    return len(selected)

print("Creating balanced fruit/vegetable dataset...")

for cls in classes:
    source = old_fruit_dir / cls
    dest = balanced_dir / cls
    count = copy_images(source, dest, cls, 100)
    print(f"{cls}: copied {count} images")

banana_dest = balanced_dir / "banana"
banana_dest.mkdir(parents=True, exist_ok=True)

banana_total = 0
for ripeness in ["green", "ripe", "overripe", "rotten"]:
    source = banana_ripeness_dir / ripeness
    count = copy_images(source, banana_dest, f"banana_{ripeness}", 50)
    banana_total += count
    print(f"banana {ripeness}: copied {count} images")

print()
print("Final balanced dataset counts:")
for folder in sorted(balanced_dir.iterdir()):
    if folder.is_dir():
        print(folder.name, len(list(folder.glob("*"))))
