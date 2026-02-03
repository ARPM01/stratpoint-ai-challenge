"""
Visualize SROIE ground truth
"""

import json
import os
import random

import cv2
import numpy as np

BASE_DIR = "data/SROIE2019/train"
IMG_DIR = os.path.join(BASE_DIR, "img")
BOX_DIR = os.path.join(BASE_DIR, "box")
ENTITIES_DIR = os.path.join(BASE_DIR, "entities")


def visualize_random_sample():
    """Visualize a random sample from the SROIE2019 dataset with ground truth boxes and entities."""
    if not os.path.exists(IMG_DIR):
        print(f"Error: Image directory not found at {IMG_DIR}")
        return

    # Get list of images
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
    if not image_files:
        print("No images found.")
        return

    # Select a random image
    filename = random.choice(image_files)
    file_id = os.path.splitext(filename)[0]

    img_path = os.path.join(IMG_DIR, filename)
    box_path = os.path.join(BOX_DIR, f"{file_id}.txt")
    entities_path = os.path.join(ENTITIES_DIR, f"{file_id}.txt")

    print(f"Visualizing Sample: {filename}")
    print("-" * 30)

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load image: {img_path}")
        return

    # Load Boxes (Ground Truth Text Locations)
    boxes = []
    if os.path.exists(box_path):
        try:
            with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(",")
                    # SROIE format: x1,y1,x2,y2,x3,y3,x4,y4,text (text can contain commas)
                    if len(parts) >= 8:
                        try:
                            coords = list(map(int, parts[:8]))
                            text = ",".join(parts[8:])
                            boxes.append({"coords": coords, "text": text})
                        except ValueError:
                            continue  # Skip malformed lines
        except Exception as e:
            print(f"Error reading box file: {e}")
    else:
        print(f"Warning: Box file not found at {box_path}")

    # Load Entities (Ground Truth Information Extraction)
    entities = {}
    if os.path.exists(entities_path):
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode entities JSON")
        except Exception as e:
            print(f"Error reading entities file: {e}")
    else:
        print(f"Warning: Entities file not found at {entities_path}")

    # Draw Boxes on Image
    # Visualization color (BGR format): Green
    BOX_COLOR = (0, 255, 0)

    for box in boxes:
        coords = box["coords"]
        pts = np.array(coords).reshape((-1, 1, 2))

        # Draw bounding box polygon
        cv2.polylines(image, [pts], isClosed=True, color=BOX_COLOR, thickness=2)

    # Print extracted entities info to console
    print("Ground Truth Entities:")
    if entities:
        for key, value in entities.items():
            print(f"  {key.capitalize()}: {value}")
    else:
        print("  No entity information available.")

    # Save output visualization
    output_filename = f"visualized_{file_id}.jpg"
    output_path = os.path.join(os.getcwd(), output_filename)

    # Optionally add entity info on the image itself (top-left corner)
    y_offset = 30
    cv2.putText(
        image,
        f"File: {filename}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    y_offset += 30

    # Save the image
    cv2.imwrite(output_path, image)
    print("-" * 30)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize_random_sample()
