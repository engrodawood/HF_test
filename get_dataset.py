import numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image
import os

def create_random_image(width, height, mode):
    """
    Create a random PIL image.
    :param width: Width of the image.
    :param height: Height of the image.
    :param mode: Color mode ('RGB' or 'L').
    :return: PIL Image object.
    """
    if mode == "RGB":
        array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # RGB image
    elif mode == "L":
        array = np.random.randint(0, 256, (height, width), dtype=np.uint8)  # Grayscale image
    else:
        raise ValueError("Unsupported mode. Use 'RGB' for color or 'L' for grayscale.")
    return Image.fromarray(array, mode)

def generate_dataset(num_samples, image_size=(128, 128)):
    """
    Generate a dataset with random 'pixel_values' and 'label'.
    :param num_samples: Number of samples in the dataset.
    :param image_size: Tuple of (width, height) for the images.
    :return: List of dictionaries with 'pixel_values' and 'label'.
    """
    width, height = image_size
    data = []
    for _ in range(num_samples):
        pixel_values = create_random_image(width, height, mode="RGB")
        label = create_random_image(width, height, mode="L")
        data.append({"pixel_values": pixel_values, "label": label})
    return data

# Parameters
num_samples_train = 10
num_samples_val = 2
num_samples_test = 2
image_size = (128, 128)

# Create datasets
train_data = generate_dataset(num_samples_train, image_size)
val_data = generate_dataset(num_samples_val, image_size)
test_data = generate_dataset(num_samples_test, image_size)

# Convert datasets to Hugging Face DatasetDict
dss = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

# Save datasets to Parquet format
output_dir = "/home/mdawood/dev/HF/test/"
os.makedirs(output_dir, exist_ok=True)

for split, dataset in dss.items():
    parquet_path = os.path.join(output_dir, f"{split}.parquet")
    dataset.to_parquet(parquet_path)

print(f"Datasets saved to {output_dir}")
