from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_convert
import torch
from tqdm import tqdm
import pandas as pd
import json


class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)
        test_file_names_path = "data/test_file_names.json"
        with open(test_file_names_path, "r") as f:
            self.test_file_names_data = json.load(f)


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.test_file_names_data["images"][idx]["file_name"]
        img_idx = self.test_file_names_data["images"][idx]["id"]
        img_path = os.path.join(self.image_folder, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "image_id": img_idx, "image_name": img_name}
    

test_dataset = TestDataset(image_folder="data/test_images/test_images", transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

ids = pd.read_csv("data/submission.csv")["ID"].to_list()
it = iter(ids)

def infer_and_save_csv(model, dataloader, device, output_csv_path):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference', unit='batch'):
            images, image_ids, image_names = batch['image'], batch['image_id'], batch['image_name']
            images = [image.to(device) for image in images]

            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = box_convert(output['boxes'], 'xyxy', 'xywh').cpu().numpy()

                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                image_id = image_ids[i].item()

                for box, label, score in zip(boxes, labels, scores):
                    if score > 0.5:
                        result = {
                            'ID': next(it),
                            'image_id': image_id,
                            'category_id': label,
                            'bbox': box.tolist(),
                            'area': box[2] * box[3],
                            'segmentation': [],
                            'iscrowd': 0,
                            'score': score
                        }
                        results.append(result)

    while True:
        try:
            result = {
                            'ID': next(it),
                            'image_id': -1,
                            'category_id': -1,
                            'bbox': "-1",
                            'area': -1,
                            'segmentation': [],
                            'iscrowd': 0,
                            'score': -1
                        }
            results.append(result)
        except StopIteration as e:
            print(e)
            break
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Inference results saved to {output_csv_path}")

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

model.load_state_dict(torch.load("trained_model.pth"))
infer_and_save_csv(model, test_loader, device, "my_submission1.csv")