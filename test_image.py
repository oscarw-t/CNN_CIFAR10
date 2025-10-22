import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# === Define the model (must match saved model) ===
class MLP_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CIFAR-10 class names
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_one().to(device)
model.load_state_dict(torch.load("mlp_one_cifar10.pth", map_location=device))
model.eval()

# === Load and preprocess image ===
img_path = "C:/Users/oscar/Desktop/KCLYEAR2/Projects/CNN_CIFAR10/data/images.jpg"
  # replace with your image file
image = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

# === Run inference ===
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"Predicted class: {classes[predicted.item()]}")
