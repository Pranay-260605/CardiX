import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict


data_dir = "Chest X-ray Data/test"  
model_path = "cnn_model_2.pth"             
batch_size = 32
image_size = 224
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(32,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),


            nn.AdaptiveAvgPool2d((1, 1)),


            nn.Flatten(),
            nn.Linear(256,128), nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = ["COVID19", "NORMAL", "PNEUMONIA"]
        self.files = []
        self.labels = []
        self.transform = transform

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = ChestXRayDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model = ChestXRayCNN(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


correct = 0
total = 0
class_correct = [0] * 4
class_total = [0] * 4

with torch.no_grad():
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1


classes = ["COVID", "NORMAL", "VIRAL PNEUMONIA"]
print(f"\n✅ Overall Accuracy: {100 * correct / total:.2f}% on {total} images")

for i in range(3):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  - {classes[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
