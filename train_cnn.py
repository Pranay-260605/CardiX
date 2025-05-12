import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

data_dir = "COVID-19_Radiography_Dataset"
batch_size = 32
epochs = 50
lr = 0.001
image_size = 224
val_split = 0.2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
        self.classes = ["COVID", "NORMAL", "VIRAL PNEUMONIA", "LUNG_OPACITY"]
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

    def __getitem__(self,idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label  
    
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = ChestXRayDataset(data_dir, transform=transform)
val_size = int(len(dataset)*val_split)
train_size = len(dataset) - val_size
train_dataset , val_dataset = random_split(dataset,[train_size,val_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size , shuffle=True)
val_loader = DataLoader(val_dataset,batch_size = batch_size )



model = ChestXRayCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = lr)
scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3,verbose=True)

print("Training the model...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    model.eval()
    correct = 0
    val_loss = 0
    total = 0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs , labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct/total
    if val_accuracy >= 95.0:
        print(f"✅ Early stopping: Val Acc reached {val_accuracy:.2f}%")
        torch.save(model.state_dict(), "cnn_model_best.pth")
        break

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")



torch.save(model.state_dict(),"cnn_model_2.pth")
