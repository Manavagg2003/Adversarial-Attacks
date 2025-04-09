import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("Standard Model Training Completed and Saved!")

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()  # Detach here to prevent gradient tracking
    return perturbed_images

def detect_adversarial(original, adversarial, threshold=0.02):
    perturbation = torch.abs(original - adversarial).mean().item()
    return perturbation > threshold  # Returns True if the change is significant

def adversarial_training(model, train_loader, epsilon=0.1, epochs=5, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial examples using FGSM
            adv_images = fgsm_attack(model, images, labels, epsilon)

            # Detect adversarial perturbations
            if detect_adversarial(images, adv_images):
                print("Adversarial example detected during training!")

            # Train on both clean and adversarial examples
            model.train()
            optimizer.zero_grad()
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            output = model(combined_images)
            loss = F.cross_entropy(output, combined_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    print("Adversarial Training Completed!")
    return model

# Define and Train Model
model = SimpleCNN().to(device)
train_model(model, trainloader, epochs=5)

# Adversarial Training for Robust Defense
adversarial_training(model, trainloader, epsilon=0.1, epochs=5, device=device)
torch.save(model.state_dict(), "models/robust_cnn_model.pth")
print("Adversarially Trained Model Saved!")