import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x16
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14x14x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x32
            
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 10)  # Direct mapping to output
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    print("\nDetailed Model Parameters:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Output Shape':<20} {'Param #'}")
    print("-" * 80)
    
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        shape = list(parameter.shape)
        print(f"{name:<40} {str(shape):<20} {param:,}")
    
    print("-" * 80)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)

    print(f"\nEpoch 1/{1}")
    print("-" * 60)

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch 1 - Batch [{i + 1}/{total_batches}], '
                  f'Loss: {running_loss/100:.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0

    return 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LightMNIST().to(device)
    
    # Display detailed parameter count
    count_parameters(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print("\nStarting training...")
    accuracy = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Final training accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
