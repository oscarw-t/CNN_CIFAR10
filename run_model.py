from clearml import Task, Logger
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mlp import MLP_one

# Initialize ClearML task
task = Task.init(
    project_name="TestProject",
    task_name="MLP_one_Evaluation",
    tags=["cnn", "pytorch", "cifar10", "evaluation"]
)

# === Setup data ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_one().to(device)
try:
    model.load_state_dict(torch.load("C:/Users/oscar/Desktop/KCLYEAR2/Projects/conv_net_raglgt/savedmodels/mlp_one_cifar10.pth", map_location=device))
except RuntimeError:
    print("runtime error loading")

model.eval()

criterion = nn.CrossEntropyLoss()
logger = task.get_logger()

# === Evaluation loop ===
correct = 0
total = 0
running_loss = 0.0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log metrics per batch to ClearML
        if i % 50 == 0:
            accuracy = 100 * correct / total
            logger.report_scalar("Evaluation", "Loss", loss.item(), iteration=i)
            logger.report_scalar("Evaluation", "Accuracy", accuracy, iteration=i)
            print(f"Batch {i}: Accuracy={accuracy:.2f}%, Loss={loss.item():.4f}")

# === Final results ===
final_accuracy = 100 * correct / total
avg_loss = running_loss / len(testloader)
print(f"\nFinal Accuracy: {final_accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")

# Log final summary to ClearML
logger.report_single_value("Final Accuracy (%)", final_accuracy)
logger.report_single_value("Average Loss", avg_loss)

# Optionally upload model again
task.upload_artifact("evaluated_model", "mlp_one_cifar10.pth")

print("✅ Evaluation complete and results uploaded to ClearML")
