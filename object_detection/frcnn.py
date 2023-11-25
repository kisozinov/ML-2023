import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import utils 
from pprint import pprint

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert
from engine import evaluate
from torchmetrics.detection import MeanAveragePrecision
from skimage import exposure
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        # Clear images w/o annotations
        print("Filtering images w/o annotations...") 
        self.annotations["images"] = [img for img in tqdm(self.annotations["images"]) if self.has_annotations(img["id"])]

    def has_annotations(self, image_id):
        return any(ann["image_id"] == image_id for ann in self.annotations["annotations"])

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        img_name = self.annotations["images"][idx]["file_name"]
        img_path = os.path.join(os.getcwd() + "/" + self.image_folder, img_name)
        
        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")
        # id изображения для маппинга в аннотации
        image_id = self.annotations["images"][idx]["id"]
        annotations = [ann for ann in self.annotations["annotations"] if ann["image_id"] == image_id]

        for i in range(len(annotations)):
            annotations[i] = {k:v for k, v in annotations[i].items() if k in ["bbox", "category_id"]}

        sample = {"image": image, "annotations": annotations}

        if self.transform:
            sample['image'] = self.transform(sample["image"])
        return sample


def collate_fn(batch):
    images = [item['image'] for item in batch]
    res = []
    for item in batch:
        d = {}
        _boxes = torch.Tensor()
        _labels = []
        for ann in item['annotations']:
            _boxes = torch.cat([_boxes, box_convert(torch.Tensor(ann["bbox"]).unsqueeze(0), "xywh", "xyxy")], dim=0)
            _labels.append(ann["category_id"])
        _labels = torch.Tensor(_labels).long()
        d["boxes"] = _boxes
        d["labels"] = _labels
        res.append(d)
    return {"images": images, "targets": res}

class LocalContrastNormalization:
    def __call__(self, sample):
        image = sample

        # Применение Local Contrast Normalization
        image = exposure.equalize_adapthist(image.numpy(), clip_limit=0.03)

        return torch.from_numpy(image)

class LocalResponseNormalization:
    def __call__(self, sample):
        image = sample

        # Применение Local Response Normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0

        # Переводим изображение в numpy array для обработки
        image_np = image.numpy()
        
        # Применяем LRN
        image_np = image_np / (bias + alpha * np.power(image_np, 2))
        image_np = image_np / np.power(1.0 + (image_np.sum(axis=0, keepdims=True) * beta), radius)

        # Возвращаем как тензор
        #print({'image': torch.from_numpy(image_np), 'annotations': sample['annotations']})
        return torch.from_numpy(image_np)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT) # pretrained=False
num_classes = 12  # Замените на количество классов в вашей задаче
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.ToTensor(),
    LocalContrastNormalization(),
    LocalResponseNormalization(),
])

train_dataset = CustomDataset(image_folder="data/train_images/train_images", annotation_file="data/train_data.json", transform=train_transform)
val_dataset = CustomDataset(image_folder="data/train_images/train_images", annotation_file="data/val_data.json", transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# print(next(iter(train_loader)))

num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch in train_loader:
            images, targets = batch['images'], batch['targets']
            
            # Обработка батча для передачи в модель
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            outputs = model(images, targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)

        lr_scheduler.step()

    average_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')
    #evaluate(model, val_loader, device=device)

    # Validation    
    model.eval()
    val_loss = 0.0
    metric = MeanAveragePrecision(iou_type="bbox")
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)', unit='batch') as pbar:
        with torch.no_grad():
            for val_batch in val_loader:
                val_images, val_targets = val_batch['images'], val_batch['targets']
                val_images = [image.to(device) for image in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                outputs = model(val_images)
            
                metric.update(outputs, val_targets)
                pbar.update(1)
    pprint(metric.compute())
    
# Сохранение обученной модели
torch.save(model.state_dict(), 'trained_model.pth')
