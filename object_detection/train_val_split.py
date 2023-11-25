import json
import random
from sklearn.model_selection import train_test_split

with open("data/usdc_train.json", "r") as f:
    data = json.load(f)

train_data, val_data = train_test_split(data["images"], test_size=0.1, random_state=42)

train_json = {"images": train_data, "annotations": data["annotations"]}
val_json = {"images": val_data, "annotations": data["annotations"]}

with open("data/train_data.json", "w") as train_file:
    json.dump(train_json, train_file)

with open("data/val_data.json", "w") as val_file:
    json.dump(val_json, val_file)