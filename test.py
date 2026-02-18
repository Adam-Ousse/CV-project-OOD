from src.data.data_loader import get_cifar100_loaders, get_ood_loader

try : 
    train_loader, val_loader, test_loader = get_cifar100_loaders()
    ood_loader = get_ood_loader('all')
    for name, loader in ood_loader.items():
        print(f'{name} loaded : {len(loader.dataset)} samples')
    print('data loaders created successfully')
except Exception as e:
    print(f'error creating data loaders: {e}')
    
    
# cifar10 loaded : 10000 samples
# tiny_imagenet loaded : 10000 samples
# mnist loaded : 10000 samples
# svhn loaded : 10000 samples
# textures loaded : 1880 samples
# data loaders created successfully