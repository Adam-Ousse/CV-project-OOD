import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import config
from src.model.model import ResNet18Classifier
from src.data.data_loader import get_cifar100_loaders
from src.nc.nc import NCMetrics


def extract_features_with_model(model, loader, device):
    """
    Extract features from all tracked layers using the model's built-in hooks.
    Returns layer_features dict and labels tensor.
    """
    model.eval()
    layer_accum = defaultdict(list)
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            _, features, layer_feats = model(inputs, return_features=True)
            for name, feat in layer_feats.items():
                layer_accum[name].append(feat.cpu())
            all_labels.append(targets)

    layer_features = {k: torch.cat(v).to(device) for k, v in layer_accum.items()}
    labels = torch.cat(all_labels).to(device)
    return layer_features, labels


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss / len(loader):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def print_nc_table(nc_metrics, nc5_metrics, epoch):
    """Print NC metrics in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"Neural Collapse Metrics - Epoch {epoch}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<15} {'Value':>12} {'Target':>12} {'Status':>20}")
    print(f"{'-' * 70}")

    # NC1
    nc1 = nc_metrics['nc1']
    status_nc1 = "✓ Collapsed" if nc1 < 0.1 else "✗ Not collapsed" if nc1 > 1.0 else "→ Collapsing"
    print(f"{'NC1 (Var)':<15} {nc1:>12.6f} {'~0':>12} {status_nc1:>20}")

    # NC2
    nc2 = nc_metrics['nc2']
    nc2_cos = nc_metrics['nc2_cos']
    status_nc2 = "✓ ETF" if nc2 < 0.1 else "✗ Not ETF" if nc2 > 1.0 else "→ Forming"
    print(f"{'NC2 (ETF)':<15} {nc2:>12.6f} {'~0':>12} {status_nc2:>20}")
    print(f"{'  └─ cos MSE':<15} {nc2_cos:>12.6f} {'~0':>12} {'':>20}")

    # NC3
    nc3 = nc_metrics['nc3']
    nc3_dist = nc_metrics['nc3_dist']
    status_nc3 = "✓ Aligned" if nc3 < 0.1 else "✗ Not aligned" if nc3 > 0.5 else "→ Aligning"
    print(f"{'NC3 (Duality)':<15} {nc3:>12.6f} {'~0':>12} {status_nc3:>20}")
    print(f"{'  └─ dist':<15} {nc3_dist:>12.6f} {'~0':>12} {'':>20}")

    # NC4
    nc4 = nc_metrics['nc4']
    status_nc4 = "✓ NCC" if nc4 > 0.99 else "✗ Not NCC" if nc4 < 0.9 else "→ Converging"
    print(f"{'NC4 (NCC)':<15} {nc4:>12.4f} {'~1.0':>12} {status_nc4:>20}")

    # NC5
    nc5_corr = nc5_metrics['nc5_correlation']
    nc5_cka = nc5_metrics['nc5_cka']
    status_nc5 = "✓ Aligned" if nc5_cka > 0.8 else "✗ Divergent" if nc5_cka < 0.3 else "→ Converging"
    print(f"{'-' * 70}")
    print(f"{'NC5 (Cross)':<15} {'':>12} {'':>12} {'':>20}")
    print(f"{'  ├─ Corr':<15} {nc5_corr:>12.6f} {'~0':>12} {'(lower better)':>20}")
    print(f"{'  └─ CKA':<15} {nc5_cka:>12.6f} {'~1.0':>12} {status_nc5:>20}")
    print(f"{'=' * 70}\n")


def train_model():
    torch.manual_seed(config.SEED)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, clean_loader = get_cifar100_loaders()

    model = ResNet18Classifier(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )

    nc_tracker = NCMetrics(num_classes=config.NUM_CLASSES, device=config.DEVICE)

    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    nc_metrics_list = []

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nepoch {epoch + 1}/{config.NUM_EPOCHS}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'train loss: {train_loss:.3f}, train acc: {train_acc:.2f}%')
        print(f'val loss: {val_loss:.3f}, val acc: {val_acc:.2f}%')

        # extract clean (no-aug) features for NC metrics
        layer_features, labels = extract_features_with_model(model, clean_loader, config.DEVICE)

        # NC1-NC4: computed on avgpool (final pre-classifier) features
        avgpool_features = layer_features['avgpool']
        nc_metrics, avgpool_means, avgpool_global_mean = nc_tracker.compute_all_metrics(
            avgpool_features, labels, model.model.fc
        )

        # NC5: cross-layer subspace orthogonality using centered class means
        layer_means_dict = {}
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            if layer_name in layer_features:
                features = layer_features[layer_name]
                means, global_mean, _, _ = nc_tracker._compute_moments(features, labels)
                centered_means = means - global_mean.unsqueeze(1)  # [p, C]
                layer_means_dict[layer_name] = centered_means

        nc5_metrics = nc_tracker.compute_nc5_cross_layer(layer_means_dict)

        nc_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'nc1': nc_metrics['nc1'],
            'nc2': nc_metrics['nc2'],
            'nc2_cos': nc_metrics['nc2_cos'],
            'nc3': nc_metrics['nc3'],
            'nc3_dist': nc_metrics['nc3_dist'],
            'nc4': nc_metrics['nc4'],
            'nc5_correlation': nc5_metrics['nc5_correlation'],
            'nc5_cka': nc5_metrics['nc5_cka'],
        }
        nc_metrics_list.append(nc_record)

        print_nc_table(nc_metrics, nc5_metrics, epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f'  ↑ new best val acc: {val_acc:.2f}% — checkpoint saved')

    # Final test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, config.DEVICE)
    print(f'\nfinal test acc: {test_acc:.2f}%')

    # Save NC metrics CSV
    nc_df = pd.DataFrame(nc_metrics_list)
    nc_csv_path = os.path.join(config.RESULTS_DIR, 'nc_metrics.csv')
    nc_df.to_csv(nc_csv_path, index=False)
    print(f'NC metrics saved to {nc_csv_path}')

    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'nc_history': nc_tracker.get_history(),
    }, os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'))

    # Clean up hooks
    model.remove_hooks()

    return model, nc_df


if __name__ == '__main__':
    train_model()