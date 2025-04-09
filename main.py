import torch
import torchvision
import torchvision.transforms as transforms
from train_model import SimpleCNN
from attacks.fgsm_attack import fgsm_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load adversarially trained model
model = SimpleCNN()
model.load_state_dict(torch.load("models/robust_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# Load dataset for testing
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Evaluate robustness
correct = 0
total = 0
epsilon = 0.1

for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    
    # Original predictions
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    # Adversarial attack
    adv_images = fgsm_attack(model, images, labels, epsilon)
    adv_outputs = model(adv_images)
    _, adv_predicted = torch.max(adv_outputs.data, 1)

    # Compare predictions
    if predicted.item() == labels.item():
        total += 1
        if adv_predicted.item() == labels.item():
            correct += 1

print(f"Robust Model Accuracy Against FGSM: {100 * correct / total:.2f}%")
