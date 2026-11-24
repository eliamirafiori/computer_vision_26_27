import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# CIFAR-10 images are 32x32; resize to 224x224 to match ResNet-18 expected input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights).to(device)

# Grab a single batch from CIFAR-10 for demonstration
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images.to(device)
sample_labels = sample_labels.to(device)

# Feature extraction: strip the final fully-connected classification layer
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
feature_extractor.eval() # Set the layer in Evaluation mode

# Use layer without back propagation
with torch.no_grad():
    raw_features = feature_extractor(sample_images)
    features = torch.flatten(raw_features, 1)

print(f"Extracted feature embedding shape: {features.shape}")
# features.cpu().numpy() can now be passed to scikit-learn classifiers (SVM, RF, etc.)

# Reload fresh model and freeze backbone weights
model = models.resnet18(weights=weights).to(device)
# Disabling model's learning parameters
for param in model.parameters():
    param.requires_grad = False

# Replace FC layer to match CIFAR-10's 10 classes
num_features = model.fc.in_features # Number of input features
NUM_CLASSES = 10
model.fc = nn.Linear(num_features, NUM_CLASSES).to(device) # Set new final layer

# Optimize only trainable parameters (model.fc)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train the FC layer for a brief demonstration run
model.train() # Set the model in Train mode (only final layer since we disabled all params)
running_loss = 0.0

for step, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if (step + 1) % 50 == 0:
        print(f"Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}")
        running_loss = 0.0

    # Optional break
    if step == 500:
        print("Stopping training early.\n")
        break

# Evaluate top-1 accuracy on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Predicted class is index of highest logit along dimension 1
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Set Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")