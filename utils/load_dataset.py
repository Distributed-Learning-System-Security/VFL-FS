"""
将imagenet数据集加载到pytorch中
"""
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
# train_dataset = torchvision.datasets.ImageFolder(
#     root='../train',
#     transform=transform)
# train_dataset_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# for image, label in train_dataset_loader:
#     print(image.shape)  # torch.Size([256, 3, 224, 224])
#     print(label.shape)  # torch.Size([256])
#     break

train_dataset = torchvision.datasets.ImageFolder(
    root='../../imagenet100/train',
    transform=transform)
train_dataset_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
for image, label in train_dataset_loader:
    print(image.shape)  # torch.Size([256, 3, 224, 224])
    print(label.shape)  # torch.Size([256])
    break

# val_dataset = torchvision.datasets.ImageFolder(
#     root='../val',
#     transform=transform)
# val_dataset_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
# for image, label in val_dataset_loader:
#     print(image.shape)  # torch.Size([256, 3, 224, 224])
#     print(label.shape)  # torch.Size([256])
#     break

val_dataset = torchvision.datasets.ImageFolder(
    root='../../imagenet100/val',
    transform=transform)
val_dataset_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
for image, label in val_dataset_loader:
    print(image.shape)  # torch.Size([256, 3, 224, 224])
    print(label.shape)  # torch.Size([256])
    break