import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt

# 1. Rebuild the Model Architecture (Must match train.py exactly)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(14 * 14 * 16, 10)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

# 2. Setup Device & Load Model Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_model.pt", map_location=device))
model.eval() # Set to evaluation mode for attacks

# 3. Load One Sample from the Test Set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Grab a single test image and its true label
image, label = test_dataset[0] 
image_tensor = image.unsqueeze(0).to(device) # Add batch dimension: [1, 1, 28, 28]
label_tensor = torch.tensor([label]).to(device)

# 4. Initialize the Foolbox Model Environment
# Bounds represent min/max pixel values after normalization (with small tolerance for float precision)
bounds = ((0.0 - 0.1307) / 0.3081 - 1e-3, (1.0 - 0.1307) / 0.3081 + 1e-3)
fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)

# 5. Run the FGSM Attack
# Epsilon defines our maximum allowed perturbation budget
EPSILON = 0.2 
fgsm_attack = fb.attacks.FGSM()
raw_fgsm, adversarial_fgsm, success_fgsm = fgsm_attack(fmodel, image_tensor, label_tensor, epsilons=EPSILON)

# 6. Extract Images & Calculate the Pixel-Level Diff
clean_np = image_tensor.squeeze().cpu().numpy()
adv_np = adversarial_fgsm.squeeze().cpu().numpy()
diff_np = adv_np - clean_np

# Get predictions
clean_pred = model(image_tensor).argmax(dim=1).item()
adv_pred = model(adversarial_fgsm).argmax(dim=1).item()

print(f"Original Label: {label} | Model Predicted: {clean_pred}")
print(f"Attack Success: {success_fgsm.item()}")
print(f"Adversarial Model Predicted: {adv_pred}")

# 7. Visualize and Save the Side-by-Side Matrix
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(clean_np, cmap='gray')
axes[0].set_title(f"Clean Image\nPred: {clean_pred}")
axes[0].axis('off')

axes[1].imshow(adv_np, cmap='gray')
axes[1].set_title(f"FGSM Adversarial\nPred: {adv_pred}")
axes[1].axis('off')

# Amplify the noise visual slightly so humans can see what changed
axes[2].imshow(np.abs(diff_np), cmap='coolwarm')
axes[2].set_title(f"Pixel Diff Matrix\n(Epsilon: {EPSILON})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("adversarial_comparison.png")
print("Visualized matrix saved as 'adversarial_comparison.png'!")
