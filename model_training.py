import subprocess
import sys

def install_required_packages():
    """Install missing dependencies automatically"""
    required_packages = [
        "ultralytics",  # Core YOLO library for model training
        "wandb",        # Training tracking tool 
        "pillow"        # Image processing support
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_required_packages()

import os
import glob
import json
import shutil
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import wandb

ANNOTATION_DIR = '/kaggle/input/face-mask-detection/annotations'
IMAGE_DIR = '/kaggle/input/face-mask-detection/images'
WORK_DIR = '/kaggle/working/face_mask_detection'

# Training parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
EPOCHS = 30
IMAGE_SIZE = 640
YOLO_VERSION = 'yolo11m.pt'
CUSTOM_MODEL_NAME = "Mask_detector.pt"  


def normalize_bounding_box(bbox, img_w, img_h):
    """Convert VOC format to YOLO normalized coordinates"""
    x_center = ((bbox[2] + bbox[0]) / 2) / img_w
    y_center = ((bbox[3] + bbox[1]) / 2) / img_h
    bbox_width = (bbox[2] - bbox[0]) / img_w
    bbox_height = (bbox[3] - bbox[1]) / img_h
    return [x_center, y_center, bbox_width, bbox_height]

def setup_directory_structure():
    """Create necessary folders for data organization"""
    dirs = [
        f"{WORK_DIR}/labels",
        f"{WORK_DIR}/dataset/train/images",
        f"{WORK_DIR}/dataset/train/labels",
        f"{WORK_DIR}/dataset/val/images",
        f"{WORK_DIR}/dataset/val/labels",
        f"{WORK_DIR}/dataset/test/images",
        f"{WORK_DIR}/dataset/test/labels",
        f"{WORK_DIR}/saved_models"  # Dedicated folder for final model
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f" All directories created. Working dir: {WORK_DIR}")

def convert_annotations_to_yolo_format():
    """Convert XML annotations to YOLO-compatible TXT files"""
    class_mapping = []
    xml_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.xml"))

    for xml_file in xml_files:
        file_id = os.path.splitext(os.path.basename(xml_file))[0]
        image_path = os.path.join(IMAGE_DIR, f"{file_id}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {file_id}.png not found - skipping")
            continue

        # Parse XML annotation file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_annotations = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                class_mapping.append(class_name)
            
            # Extract bounding box coordinates
            bbox = obj.find("bndbox")
            voc_bbox = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text)
            ]
            
            # Convert to YOLO format
            yolo_bbox = normalize_bounding_box(voc_bbox, img_width, img_height)
            yolo_line = f"{class_mapping.index(class_name)} {' '.join(map(str, yolo_bbox))}"
            yolo_annotations.append(yolo_line)

        # Save YOLO annotations to file
        if yolo_annotations:
            with open(f"{WORK_DIR}/labels/{file_id}.txt", "w") as f:
                f.write("\n".join(yolo_annotations))

    # Save class mapping for reference
    with open(f"{WORK_DIR}/labels/class_names.json", "w") as f:
        json.dump(class_mapping, f)
    print(f" Annotation conversion complete. Classes: {class_mapping}")

def split_dataset_into_sets():
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    random.shuffle(image_ids)
    
    total = len(image_ids)
    train_count = int(total * TRAIN_SPLIT)
    val_count = int(total * VAL_SPLIT)
    test_count = total - train_count - val_count

    def copy_to_set(set_name, ids):
        for file_id in ids:
            # Copy image file
            src_img = os.path.join(IMAGE_DIR, f"{file_id}.png")
            dst_img = os.path.join(WORK_DIR, "dataset", set_name, "images", f"{file_id}.png")
            
            # Copy label file (if exists)
            src_label = os.path.join(WORK_DIR, "labels", f"{file_id}.txt")
            dst_label = os.path.join(WORK_DIR, "dataset", set_name, "labels", f"{file_id}.txt")

            if os.path.exists(src_label):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_label, dst_label)

    copy_to_set("train", image_ids[:train_count])
    copy_to_set("val", image_ids[train_count:train_count+val_count])
    copy_to_set("test", image_ids[train_count+val_count:])
    print(f" Dataset split: Train({train_count}) | Val({val_count}) | Test({test_count})")

def create_yolo_config_file():
    config_content = f"""
# YOLO11 Face Mask Detection Config
train: {WORK_DIR}/dataset/train/images
val: {WORK_DIR}/dataset/val/images

# Class configuration
nc: 3
names: ["with_mask", "mask_weared_incorrect", "without_mask"]
    """.strip()
    
    config_path = f"{WORK_DIR}/dataset/mask_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f" YOLO config file created at: {config_path}")
    return config_path

def train_yolo_mask_detection_model(config_path):
    wandb.init(mode="disabled")
    
    # Initialize YOLO model with pretrained weights
    model = YOLO(YOLO_VERSION)
    
    # Start model training with key parameters
    print("\n=== Starting Model Training ===")
    model.train(
        data=config_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=0,  # Use Kaggle GPU 
        save=True,               # Save best model 
        save_period=1,           # Save model every epoch 
        exist_ok=True,           # Overwrite existing training results
        project=f"{WORK_DIR}/training_results",
        name="mask_detection_run"
    )
    
 
    final_model_path = os.path.join(WORK_DIR, "saved_models", CUSTOM_MODEL_NAME)
    best_model_original = f"{WORK_DIR}/training_results/mask_detection_run/weights/best.pt"
    last_model_original = f"{WORK_DIR}/training_results/mask_detection_run/weights/last.pt"

    if os.path.exists(best_model_original):
        shutil.copy2(best_model_original, final_model_path)
        print(f"\n Training Complete!")
        print(f" Model saved as custom name: {final_model_path}")
        print(f"\n In Kaggle, find your model at: /kaggle/working/face_mask_detection/saved_models/{CUSTOM_MODEL_NAME}")
        return final_model_path

    elif os.path.exists(last_model_original):
        shutil.copy2(last_model_original, final_model_path)
        print(f"\n Best model not found - using last epoch model")
        print(f" Model saved as custom name: {final_model_path}")
        return final_model_path
    else:
        print("\n Error: No model files found! Training may have failed.")
        return None


if __name__ == "__main__":
    setup_directory_structure()
    convert_annotations_to_yolo_format()
    split_dataset_into_sets()
    config_file = create_yolo_config_file()
    trained_model_path = train_yolo_mask_detection_model(config_file)
    print("\n=== All Processes Completed Successfully ===")
    if trained_model_path:
        print(f" Model successfully saved as: {trained_model_path}")
        # Show final model filename for quick reference
        print(f" Final model file: {os.path.basename(trained_model_path)}")
    else:
        print(" Model saving failed!")