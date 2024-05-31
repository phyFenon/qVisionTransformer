import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit import VisionTransformer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# Define validation function in the case of overfitting
def validate(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy



# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
])
# Load CIFAR-10 train and test datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



# Instantiate ViT model
d_model = 128
num_heads = 8
mlp_hidden = 256 # 2*d_model
num_layers = 8
patch_size = 4
image_size = 32
num_classes = 10
dropout_rate = 0.2
batch_size = 64
####################

# Create DataLoader for CIFAR-10 train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Instantiate ViT model
vit_model = VisionTransformer(d_model, num_heads, mlp_hidden, num_layers, patch_size, image_size, num_classes, dropout_rate)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=0.01,weight_decay=1e-4)
# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # LR decreases by a factor of 0.5 every 10 epochs


# Create figure and axis outside the loop
plt.ion()
fig, ax1 = plt.subplots(figsize=(8, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Accuracy (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
plt.grid(True)
plt.title('Training Loss and Accuracy vs. Epoch')


# Early stopping parameters
#为了防止模型在训练集上的过拟合，在测试验证集上如果val_loss累计超过十次没有降低，则停止训练
patience = 10
early_stop = False
best_val_loss = np.inf
counter = 0
#######



# Training loop with early stopping
num_epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    t_start = time.time()
    vit_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = vit_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    
    # Validate the model
    val_loss, val_accuracy = validate(vit_model, criterion, test_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step()  # Adjust learning rate
    t_end = time.time()
   
    print('[Epoch %d]/[Epoch %d] Train Loss: %.3f, Train Accuracy: %.2f %%, Val Loss: %.3f, Val Accuracy: %.2f %%, Time: %.2f s' % (epoch + 1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, t_end - t_start))
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping at epoch', epoch+1)
            early_stop = True
            break

    if early_stop:
        break
    
    # Update the dynamic plot
    ax1.plot(range(1, epoch + 2), train_losses, 'o-', label='Train_Loss', color=color1)
    ax2.plot(range(1, epoch + 2), train_accuracies, '*-', label='Train_Accuracy', color=color2)
    plt.pause(0.1)
    plt.ioff()

plt.savefig('train.png')
# Save the trained model parameters
torch.save(vit_model.state_dict(), 'vit_model.pth')
print('Finished Training')