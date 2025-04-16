
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import csv
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

epsilon = 0.1
correct = 0
total = 0
clean_correct = 0
adv_misclassified = []

for images, labels in testloader:
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    if predicted.item() == labels.item():
        clean_correct += 1

    adv_images = fgsm_attack(model, images, labels, epsilon)
    adv_outputs = model(adv_images)
    _, adv_predicted = torch.max(adv_outputs.data, 1)

    if predicted.item() == labels.item():
        total += 1
        if adv_predicted.item() == labels.item():
            correct += 1
        else:
            adv_misclassified.append((images.cpu(), adv_images.cpu(), labels.item(), adv_predicted.item()))

# Results
clean_acc = 100 * clean_correct / len(testloader)
adv_acc = 100 * correct / total

print(f"After attack: {clean_acc:.2f}%")
print(f"Robust Accuracy Against FGSM (ε={epsilon}): {adv_acc:.2f}%")

# Save results to CSV
with open("fgsm_accuracy_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "After Attack (%)", "After defense (%)"])
    writer.writerow([epsilon, clean_acc, adv_acc])

# Plot bar graph of clean vs FGSM accuracy
plt.figure(figsize=(6, 4))
plt.bar(["After attack", "After defense"], [clean_acc, adv_acc], color=["green", "red"])
plt.ylabel("Accuracy (%)")
plt.title(f"Model Robustness to FGSM Attack (ε={epsilon})")
plt.grid(axis='y')
plt.savefig("fgsm_accuracy_comparison.png")
plt.show()

# Visualize 5 misclassified adversarial examples
if adv_misclassified:
    plt.figure(figsize=(10, 4))
    for i, (orig, adv, label, adv_label) in enumerate(adv_misclassified[:5]):
        plt.subplot(2, 5, i+1)
        plt.imshow(TF.to_pil_image(orig[0]))
        plt.title(f"Clean: {label}")
        plt.axis('off')
        
        plt.subplot(2, 5, i+6)
        plt.imshow(TF.to_pil_image(adv[0]))
        plt.title(f"FGSM: {adv_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("fgsm_misclassifications.png")
    plt.show()
