import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Setup Hyperparameters & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.01

# 2. Load the Dataset (MNIST Digits)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST scaling
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Build a Simple, Boring Classifier
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

# 4. Initialize Training Components
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. The Training Loop
print(f"Starting training on {device}...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()       # Clear old gradients
        output = model(data)        # Forward pass
        loss = criterion(output, target) # Calculate loss
        loss.backward()             # Backward pass (calculate new gradients)
        optimizer.step()            # Update model weights
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} completed. Avg Loss: {total_loss/len(train_loader):.4f}")

# 6. Save the Weights
torch.save(model.state_dict(), "mnist_model.pt")
print("Model successfully trained and saved as 'mnist_model.pt'!")
