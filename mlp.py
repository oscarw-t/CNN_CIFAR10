#mlp one

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
      
        return x
    

import torchvision.transforms as transforms

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
    model = MLP_one().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)#lr is learning rate

    for epoch in range(50):#epoch is trained once over set of data
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"[{epoch+1}, {i+1}] loss: {loss.item():.3f}")
    torch.save(model.state_dict(), "mlp_one_cifar10.pth")
    print("Training complete ✅")

if __name__ == "__main__":
    main()