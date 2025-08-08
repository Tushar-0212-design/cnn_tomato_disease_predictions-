import os
import shutil
from sklearn.model_selection import train_test_split

# Source and destination paths
dataset_path = r"D:/Project/Tomato Augmantation"
train_path = r"D:/Project/Tomato_Data/train"
val_path = r"D:/Project/Tomato_Data/val"
test_path = r"D:/Project/Tomato_Data/test"

# Create class-wise folders in train, val, and test directories
disease_classes = os.listdir(dataset_path)
for cls in disease_classes:
    for base_path in [train_path, val_path, test_path]:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

# Function to split and move images for each class
def split_and_save(class_name):
    source_folder = os.path.join(dataset_path, class_name)
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for img in train:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_path, class_name, img))
    for img in val:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_path, class_name, img))
    for img in test:
        shutil.copy(os.path.join(source_folder, img), os.path.join(test_path, class_name, img))

# Run for each class
for cls in disease_classes:
    split_and_save(cls)

print(" Dataset split into Train (70%), Val (15%), Test (15%) successfully!")



##########   rename  ######## 

def rename_images(directory, label):
    images = sorted(os.listdir(directory))
    for index, img in enumerate(images):
        ext = os.path.splitext(img)[1]
        new_name = f"{label}_{index}{ext}"
        os.rename(os.path.join(directory, img), os.path.join(directory, new_name))

# Rename images in train, val, test sets
for split in ["train", "val", "test"]:
    split_dir = os.path.join(r"D:/Project/Tomato_Data", split)
    for cls in os.listdir(split_dir):
        rename_images(os.path.join(split_dir, cls), cls)

print(" Image renaming complete!")
