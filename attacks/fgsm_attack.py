import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import SimpleCNN  # Ensure model is imported
from utils.visualize import show_images  # Ensure visualization is imported


# ✅ Define FGSM Attack First
def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed_images = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed_images, 0, 1)

# ✅ Load Dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True)

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model.eval()

# ✅ Get a Batch of Images
dataiter = iter(testloader)
original_images, labels = next(dataiter)
original_images, labels = original_images.to(device), labels.to(device)

# ✅ Generate Adversarial Images
epsilon = 0.1  # Attack strength
adversarial_images = fgsm_attack(model, original_images, labels, epsilon)  # No more error!

# ✅ Display Images
show_images(original_images, adversarial_images)
