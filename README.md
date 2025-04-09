
# 🛡️ Adversarial Attacks in Deep Learning

This project explores adversarial attacks and defenses on deep learning models, specifically targeting image classification tasks. The goal is to demonstrate how small, imperceptible perturbations to input images can mislead well-trained models — and how to build resilience against such attacks.

## 🚀 Project Highlights

✅ Implemented Fast Gradient Sign Method (FGSM) for generating adversarial examples.  
🧠 Trained CNN on CIFAR-10 dataset using PyTorch.  
🔍 Evaluated the effect of varying perturbation strengths (ε).  
🔐 Implemented and tested basic defense mechanisms.  
📈 Analyzed model accuracy pre- and post-attack & defense.  

## 📁 Folder Structure

```
Adversarial-Attacks/
├── data/                         # CIFAR-10 dataset (not uploaded due to size)
├── models/                       # Trained model checkpoints
├── notebooks/                    # Jupyter notebooks for visualization & experimentation
├── src/
│   ├── train.py                  # CNN training code
│   ├── attack.py                 # FGSM attack code
│   └── defense.py                # Defense techniques
├── utils/                        # Helper functions (e.g., plotters, loaders)
├── results/                      # Visualizations and evaluation metrics
├── .gitattributes                # Git LFS tracking
├── .gitignore
└── README.md
```

## ⚙️ Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Manavagg2003/Adversarial-Attacks.git
cd Adversarial-Attacks
```

2. **Make sure you're using Python 3.8+ with PyTorch and matplotlib**

3. **Download CIFAR-10 Dataset**  
Either modify the code to auto-download or manually place it under `data/`.

## 🧪 Usage

🔧 Train the Model  
```bash
python src/train.py
```

⚔️ Run Adversarial Attack  
```bash
python src/attack.py --epsilon 0.1
```

🛡️ Apply Defense Techniques  
```bash
python src/defense.py
```

## 🧠 Concepts Explored

- Adversarial Examples  
- Gradient-Based Attacks (FGSM)  
- Model Robustness  
- Defense Strategies (e.g., input preprocessing, adversarial training)  

## 📚 References

- [Explaining and Harnessing Adversarial Examples (Goodfellow et al.)](https://arxiv.org/abs/1412.6572)  
- [PyTorch Documentation](https://pytorch.org)  
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  

## 👨‍💻 Author

**Manav Aggarwal**  
GitHub: [Manavagg2003](https://github.com/Manavagg2003)

## 📜 License

This project is open-source and available under the MIT License.
