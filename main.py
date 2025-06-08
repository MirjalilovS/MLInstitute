# Import necessary libraries
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Download and preprocess the MNIST training dataset
training_data = datasets.MNIST(
    root="data",  # Directory to store the dataset
    train=True,  # Load the training set
    download=True,  # Download the dataset if not already present
    transform=transforms.Compose([  # Apply transformations
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std
    ]),
)

# Download and preprocess the MNIST test dataset
test_data = datasets.MNIST(
    root="data",  # Directory to store the dataset
    train=False,  # Load the test set
    download=True,  # Download the dataset if not already present
    transform=transforms.Compose([  # Apply transformations
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std
    ]),
)

# Define a mapping for labels (digits 0-9)
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # preserves 28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # preserves 28×28
        self.pool  = nn.MaxPool2d(2, 2)                          # down to 14×14
        #needs to be fixed, calling dropout on 2d tensors will error out in future versions
        self.dropout = nn.Dropout2d(0.25)
        
        # Classifier head
        self.fc1   = nn.Linear(64 * 14 * 14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = SimpleCNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    total_confidence = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            probs, preds = torch.softmax(pred, dim=1).max(dim=1)
            accuracy    += (preds == y).sum().item()
            total_confidence += probs.sum().item()
    test_loss /= num_batches
    accuracy /= size
    total_confidence /= size
    print(f"Test Error: \n"
          f"  Accuracy: {100*accuracy:>0.1f}%, "
          f"Avg loss: {test_loss:>8f}, "
          f"Avg confidence: {total_confidence:.2%}\n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# After training loop finishes
torch.save(model.state_dict(), "mnist_cnn.pth")
