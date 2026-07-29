import os
import shutil
from pathlib import Path

project_dir = Path("/home/rubi/Documents/freshness_alert_project")

banana_ripeness_dir = project_dir / "dataset" / "banana_ripeness"
fruit_banana_dir = project_dir / "dataset" / "fruit_veg" / "banana"

fruit_banana_dir.mkdir(parents=True, exist_ok=True)

ripeness_classes = ["green", "ripe", "overripe", "rotten"]

copied_total = 0

for cls in ripeness_classes:
    source_dir = banana_ripeness_dir / cls

    if not source_dir.exists():
        print(f"Missing folder: {source_dir}")
        continue

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(source_dir.glob(ext))

    print(f"{cls}: found {len(images)} images")

    for i, img_path in enumerate(images[:500]):
        new_name = f"banana_{cls}_{i}_{img_path.name}"
        dest_path = fruit_banana_dir / new_name

        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
            copied_total += 1

print()
print(f"Copied total new banana images: {copied_total}")
print(f"Final banana folder: {fruit_banana_dir}")
print(f"Final banana image count: {len(list(fruit_banana_dir.glob('*')))}")
