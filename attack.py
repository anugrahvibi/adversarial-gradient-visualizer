import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Model Architecture Definition
# -----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST digit classification.
    Must match the architecture used in train.py.
    """
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


def main():
    # -------------------------------------------------------------------------
    # 2. Setup Device & Load Trained Model Weights
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("mnist_model.pt", map_location=device))
    model.eval()  # Set to evaluation mode for deterministic attack behavior

    # -------------------------------------------------------------------------
    # 3. Load MNIST Test Dataset & Select a Correctly-Classified Sample
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initialize Foolbox Model Environment
    # Bounds represent normalized pixel range (0 to 1 transformed by MNIST mean & std)
    bounds = ((0.0 - 0.1307) / 0.3081 - 1e-3, (1.0 - 0.1307) / 0.3081 + 1e-3)
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)

    # Attack Hyperparameters
    EPSILON = 0.2
    fgsm_attack = fb.attacks.FGSM()
    pgd_attack = fb.attacks.LinfPGD(steps=40, rel_stepsize=0.2)

    # Search for a correctly classified sample where attacks demonstrate clear adversarial perturbation
    image_tensor, label_tensor, clean_pred, clean_logits = None, None, None, None
    for idx in range(len(test_dataset)):
        img, lbl = test_dataset[idx]
        img_t = img.unsqueeze(0).to(device)
        lbl_t = torch.tensor([lbl]).to(device)
        with torch.no_grad():
            out = model(img_t)
            pred = out.argmax(dim=1).item()
        if pred == lbl:
            # Check if attack induces perturbation on this sample
            _, _, fgsm_succ = fgsm_attack(fmodel, img_t, lbl_t, epsilons=EPSILON)
            if fgsm_succ.item():
                image_tensor = img_t
                label_tensor = lbl_t
                clean_pred = pred
                clean_logits = out
                print(f"Selected Test Sample Index: {idx} | Ground Truth Label: {lbl}")
                break

    clean_probs = torch.softmax(clean_logits, dim=1)[0]
    clean_conf = clean_probs[clean_pred].item() * 100

    # -------------------------------------------------------------------------
    # 4. Execute FGSM Attack (One-step Fast Gradient Sign Method)
    # -------------------------------------------------------------------------
    _, fgsm_adv_tensor, fgsm_success = fgsm_attack(
        fmodel, image_tensor, label_tensor, epsilons=EPSILON
    )

    with torch.no_grad():
        fgsm_logits = model(fgsm_adv_tensor)
    fgsm_pred = fgsm_logits.argmax(dim=1).item()
    fgsm_conf = torch.softmax(fgsm_logits, dim=1)[0][fgsm_pred].item() * 100

    # -------------------------------------------------------------------------
    # 5. Execute PGD Attack (Projected Gradient Descent - Linf Bounded)
    # -------------------------------------------------------------------------
    _, pgd_adv_tensor, pgd_success = pgd_attack(
        fmodel, image_tensor, label_tensor, epsilons=EPSILON
    )

    with torch.no_grad():
        pgd_logits = model(pgd_adv_tensor)
    pgd_pred = pgd_logits.argmax(dim=1).item()
    pgd_conf = torch.softmax(pgd_logits, dim=1)[0][pgd_pred].item() * 100

    # -------------------------------------------------------------------------
    # 6. Extract NumPy Arrays & Calculate Pixel-Level Differences
    # -------------------------------------------------------------------------
    clean_np = image_tensor.squeeze().cpu().numpy()
    fgsm_np = fgsm_adv_tensor.squeeze().cpu().numpy()
    pgd_np = pgd_adv_tensor.squeeze().cpu().numpy()

    fgsm_diff_np = fgsm_np - clean_np
    pgd_diff_np = pgd_np - clean_np

    # -------------------------------------------------------------------------
    # 7. Terminal Summary & Adversarial Strength Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("           ADVERSARIAL ATTACK COMPARISON SUMMARY           ")
    print("=" * 65)
    print(f"Ground Truth Label       : {label_tensor.item()}")
    print(f"Clean Model Prediction   : Class {clean_pred} (Confidence: {clean_conf:.2f}%)")
    print("-" * 65)
    print(f"FGSM Attack (eps={EPSILON})")
    print(f"  - Attack Success       : {fgsm_success.item()}")
    print(f"  - Adversarial Prediction: Class {fgsm_pred} (Confidence: {fgsm_conf:.2f}%)")
    print(f"  - Max L_inf Perturbation: {np.abs(fgsm_diff_np).max():.4f}")
    print("-" * 65)
    print(f"PGD Attack (eps={EPSILON}, steps=40, rel_stepsize=0.01)")
    print(f"  - Attack Success       : {pgd_success.item()}")
    print(f"  - Adversarial Prediction: Class {pgd_pred} (Confidence: {pgd_conf:.2f}%)")
    print(f"  - Max L_inf Perturbation: {np.abs(pgd_diff_np).max():.4f}")
    print("=" * 65)
    print("\nAdversarial Strength Insight:")
    print("  * FGSM (Fast Gradient Sign Method) performs a single one-step update")
    print("    along the gradient sign direction. It is fast but can easily overshoot")
    print("    or get stuck in suboptimal directions on non-linear loss landscapes.")
    print("  * PGD (Projected Gradient Descent) performs 40 iterative gradient steps,")
    print("    projecting perturbations back onto the L_inf epsilon ball at each step.")
    print("    This fine-grained iterative refinement makes PGD a significantly stronger")
    print("    and more reliable adversary for robustness evaluations.\n")

    # -------------------------------------------------------------------------
    # 8. Generate 1x5 Matplotlib Visual Matrix Subplot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    # Column 1: Clean Image
    axes[0].imshow(clean_np, cmap='gray')
    axes[0].set_title(f"1. Clean Image\nPred: {clean_pred} ({clean_conf:.1f}%)", fontsize=10)
    axes[0].axis('off')

    # Column 2: FGSM Adversarial Image
    axes[1].imshow(fgsm_np, cmap='gray')
    axes[1].set_title(f"2. FGSM Adversarial\nPred: {fgsm_pred} ({fgsm_conf:.1f}%)", fontsize=10)
    axes[1].axis('off')

    # Column 3: FGSM Absolute Pixel Noise Matrix
    axes[2].imshow(np.abs(fgsm_diff_np), cmap='coolwarm')
    axes[2].set_title(f"3. FGSM Noise Diff\n(L_inf eps={EPSILON})", fontsize=10)
    axes[2].axis('off')

    # Column 4: PGD Adversarial Image
    axes[3].imshow(pgd_np, cmap='gray')
    axes[3].set_title(f"4. PGD Adversarial\nPred: {pgd_pred} ({pgd_conf:.1f}%)", fontsize=10)
    axes[3].axis('off')

    # Column 5: PGD Absolute Pixel Noise Matrix
    axes[4].imshow(np.abs(pgd_diff_np), cmap='coolwarm')
    axes[4].set_title(f"5. PGD Noise Diff\n(Steps=40, eps={EPSILON})", fontsize=10)
    axes[4].axis('off')

    plt.tight_layout()
    output_filename = "fgsm_vs_pgd_comparison.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    print(f"Visualized matrix saved successfully as '{output_filename}'!")


if __name__ == "__main__":
    main()
