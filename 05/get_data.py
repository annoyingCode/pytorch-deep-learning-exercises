
from pathlib import Path

import os
import requests
import zipfile

# Setup path to data folder
data_path = Path("data/")
food_data_path = data_path / "pizza_steak_sushi"

# Check if the food data already exists. If not, download it.
if food_data_path.is_dir():
    print(f"{food_data_path} already exists. Downloading skipped!")
    exit(0)
else:
    print(f"Did not find {food_data_path}. Creating one...")
    food_data_path.mkdir(parents=True, exist_ok=True)
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        print("Downloading food data...")
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)
    
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping food data...")
        zip_ref.extractall(food_data_path)

# Remove the zip file after downloading and extracting
print("Removing zip file: pizza_steak_sushi.zip")
os.remove(data_path / "pizza_steak_sushi.zip")

print(f"Food data successfully downloaded and extracted to {food_data_path}")
