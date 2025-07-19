import torch
import torchvision.transforms as transforms 
from torchvision import datasets
import numpy as np 
from traintest import train
import build_model as build

torch.random.manual_seed(1)

# Dataset Config -------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {
        'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std)
                ]), 
        'val': transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Resize((224, 224), antialias=None),
                    transforms.Normalize(mean, std)
                ])
    }

status = False
# Todo: Train on CIFAR10
train_dataset = datasets.CIFAR10(
                root='./datasets/torch_cifar10/', 
                train=True, 
                transform=data_transform['train'], 
                download=True)
val_dataset = datasets.CIFAR10(
                root='./datasets/torch_cifar10/', 
                train=False, 
                transform=data_transform['val'], 
                download=True)

# Todo: Train on CIFAR100
train_dataset = datasets.CIFAR100(
                root='./datasets/torch_cifar100/', 
                train=True, 
                transform=data_transform['train'], 
                download=True)
val_dataset = datasets.CIFAR100(
                root='./datasets/torch_cifar100/', 
                train=False, 
                transform=data_transform['val'], 
                download=True)

batch_size = 12
train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)

val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=8, 
                pin_memory=True)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)
    datasets = 'cifar10'  # or 'cifar100'

    if datasets == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    swin_type = 'tiny'

    model = build.buildSparseSwin(
        image_resolution=224,
        swin_type='tiny',
        num_classes=num_classes,
        ltoken_num=49,
        ltoken_dims=256,
        num_heads=16,
        qkv_bias=True,
        lf=2,
        attn_drop_prob=.0,
        lin_drop_prob=.0,
        freeze_12=False,
        device=device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    train(
        train_loader, 
        swin_type, 
        datasets, 
        epochs, 
        model, 
        lf, 
        ltoken_num,
        optimizer, 
        criterion, 
        device, 
        show_per=show_per,
        reg_type=reg_type, 
        reg_lambda=reg_lambda, 
        validation=val_loader)
