import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

from train_model import SimpleCNN
from attacks.fgsm_attack import fgsm_attack
from defenses.adversarial_training import adversarial_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# Sample adversarial attack
image, label = trainset[0]
image = image.unsqueeze(0).to(device)
label = torch.tensor([label]).to(device)

adv_image = fgsm_attack(model, image, label, epsilon=0.1)
print("Adversarial Attack Completed!")

# Detect if adversarial attack is effective
perturbation = torch.abs(image - adv_image).mean().item()
if perturbation > 0.02:
    print("Adversarial Attack Detected! Training with defense...")

# Adversarial Training
adversarial_training(model, trainloader, epsilon=0.1, epochs=5, device=device)

# Save new robust model
torch.save(model.state_dict(), "models/robust_cnn_model.pth")
print("Adversarially Trained Model Saved!")
