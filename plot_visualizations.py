import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import config
from src.model.model import ResNet18Classifier
from src.data.data_loader import get_cifar100_loaders, get_ood_loader

sns.set_theme(style='whitegrid', font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def extract_features(model, loader, device):
    model.eval()
    all_features, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits, features, _ = model(inputs, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_logits.append(logits.cpu())
    return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_logits)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # load model
    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'),
                      map_location=config.DEVICE)
    model = ResNet18Classifier(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    C = config.NUM_CLASSES
    W = model.model.fc.weight.detach().cpu().numpy()  # [C, p]

    # extract train features for NC metrics (class means, covariances, etc.)
    train_loader, val_loader, test_loader, clean_loader = get_cifar100_loaders()
    print('extracting train features...')
    train_feat, train_lab, _ = extract_features(model, clean_loader, config.DEVICE)
    train_feat_np = train_feat.numpy()
    train_lab_np = train_lab.numpy()

    # class means [C, p] and global mean [p]
    class_means = np.array([
        train_feat_np[train_lab_np == c].mean(axis=0) for c in range(C)
    ])
    global_mean = train_feat_np.mean(axis=0)
    centered_means = class_means - global_mean  # [C, p]

    # extract test features for PCA / OOD plots
    print('extracting test features...')
    test_feat, test_lab, _ = extract_features(model, test_loader, config.DEVICE)
    test_feat_np = test_feat.numpy()
    test_lab_np = test_lab.numpy()

    # subset: 10 classes, 10 points each for clean cluster plots
    rng = np.random.RandomState(42)
    show_classes = rng.choice(C, 10, replace=False)
    show_classes.sort()
    subset_idx = []
    for c in show_classes:
        c_idx = np.where(test_lab_np == c)[0]
        subset_idx.append(rng.choice(c_idx, min(10, len(c_idx)), replace=False))
    subset_idx = np.concatenate(subset_idx)
    sub_feat = test_feat_np[subset_idx]
    sub_lab = test_lab_np[subset_idx]

    # =========================================================================
    # 1. class mean pairwise distances heatmap
    # =========================================================================
    print('plotting class mean distances...')
    dist_matrix = np.linalg.norm(
        centered_means[:, None, :] - centered_means[None, :, :], axis=2
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap='viridis', square=True, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'L2 distance'})
    ax.set_title('Pairwise distance between centered class means')
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    fig.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_distances.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_distances.png'))
    plt.close(fig)

    # =========================================================================
    # 2. within-class variance per class
    # =========================================================================
    print('plotting within-class variance...')
    within_var = np.array([
        np.mean(np.sum((train_feat_np[train_lab_np == c] - class_means[c]) ** 2, axis=1))
        for c in range(C)
    ])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(C), np.sort(within_var)[::-1], color='steelblue', width=1.0, edgecolor='none')
    ax.set_xlabel('Class (sorted by variance)')
    ax.set_ylabel(r'$\frac{1}{n_c}\sum \|h - \mu_c\|^2$')
    ax.set_title('Within-class variance per class (sorted)')
    ax.set_xlim(-0.5, C - 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'within_class_variance.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'within_class_variance.png'))
    plt.close(fig)

    # =========================================================================
    # 3. cosine similarity: classifier weights vs centered class means
    # =========================================================================
    print('plotting NC3 alignment...')
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    M_norm = centered_means / (np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(W_norm * M_norm, axis=1)  # [C]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cos_sim, bins=40, color='seagreen', edgecolor='black', alpha=0.85)
    ax.axvline(cos_sim.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'mean = {cos_sim.mean():.3f}')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, label='perfect = 1.0')
    ax.set_xlabel(r'$\cos(\mathbf{w}_c,\, \mu_c - \mu_G)$')
    ax.set_ylabel('Count')
    ax.set_title('NC3: Classifier weight / class mean alignment')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'nc3_cosine_alignment.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'nc3_cosine_alignment.png'))
    plt.close(fig)

    # =========================================================================
    # 4. Gram matrix of normalized class means (NC2 — ETF structure)
    # =========================================================================
    print('plotting NC2 Gram matrix...')
    G = M_norm @ M_norm.T  # [C, C]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # actual
    sns.heatmap(G, cmap='RdBu_r', center=0, square=True, ax=axes[0],
                xticklabels=False, yticklabels=False, vmin=-0.15, vmax=1.0,
                cbar_kws={'label': 'cosine similarity'})
    axes[0].set_title('Actual Gram matrix')

    # ideal ETF
    G_ideal = np.full((C, C), -1.0 / (C - 1))
    np.fill_diagonal(G_ideal, 1.0)
    sns.heatmap(G_ideal, cmap='RdBu_r', center=0, square=True, ax=axes[1],
                xticklabels=False, yticklabels=False, vmin=-0.15, vmax=1.0,
                cbar_kws={'label': 'cosine similarity'})
    axes[1].set_title('Ideal ETF Gram matrix')
    fig.suptitle('NC2: Normalized class means — actual vs ETF', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'nc2_gram_matrix.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'nc2_gram_matrix.png'))
    plt.close(fig)

    # =========================================================================
    # 5. class mean norms (should be equal for perfect NC2)
    # =========================================================================
    print('plotting class mean norms...')
    mean_norms = np.linalg.norm(centered_means, axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(C), np.sort(mean_norms)[::-1], color='coral', width=1.0, edgecolor='none')
    ax.axhline(mean_norms.mean(), color='black', linestyle='--', linewidth=1,
               label=f'mean = {mean_norms.mean():.2f}, std = {mean_norms.std():.2f}')
    ax.set_xlabel('Class (sorted)')
    ax.set_ylabel(r'$\|\mu_c - \mu_G\|$')
    ax.set_title('NC2: Class mean norms (should be equal)')
    ax.legend()
    ax.set_xlim(-0.5, C - 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_norms.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_norms.png'))
    plt.close(fig)

    # =========================================================================
    # 6. PCA 2D — 10 ID classes colored by class
    # =========================================================================
    print('plotting PCA 2D (10 ID classes)...')
    pca2 = PCA(n_components=2)
    feat_2d = pca2.fit_transform(sub_feat)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(feat_2d[:, 0], feat_2d[:, 1],
                         c=sub_lab, cmap='tab10', s=30, alpha=0.7)
    # plot class means projected for the 10 classes
    means_2d = pca2.transform(class_means[show_classes])
    ax.scatter(means_2d[:, 0], means_2d[:, 1], c='black', s=60, marker='x', linewidths=1.5,
               label='class means', zorder=5)
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('PCA 2D — 10 CIFAR-100 classes (10 pts each)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_id_classes.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_id_classes.png'))
    plt.close(fig)

    # =========================================================================
    # 7. PCA 3D — 10 ID classes
    # =========================================================================
    print('plotting PCA 3D (10 ID classes)...')
    pca3 = PCA(n_components=3)
    feat_3d = pca3.fit_transform(sub_feat)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feat_3d[:, 0], feat_3d[:, 1], feat_3d[:, 2],
               c=sub_lab, cmap='tab10', s=20, alpha=0.6)
    means_3d = pca3.transform(class_means[show_classes])
    ax.scatter(means_3d[:, 0], means_3d[:, 1], means_3d[:, 2],
               c='black', s=50, marker='x', linewidths=1.5)
    ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})')
    ax.set_title('PCA 3D — 10 CIFAR-100 classes')
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_3d_id_classes.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_3d_id_classes.png'))
    plt.close(fig)

    # =========================================================================
    # 8. PCA 2D — ID test vs each OOD dataset
    # =========================================================================
    ood_datasets = {
        'cifar10': 'CIFAR-10 (near)', 'tiny_imagenet': 'Tiny ImageNet (near)',
        'mnist': 'MNIST (far)', 'svhn': 'SVHN (far)', 'textures': 'Textures (far)',
    }
    all_ood_loaders = get_ood_loader('all')

    # subsample ID test
    n_id_show = min(2000, len(test_feat_np))
    id_idx = rng.choice(len(test_feat_np), n_id_show, replace=False)
    id_sub = test_feat_np[id_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for i, (key, display) in enumerate(ood_datasets.items()):
        print(f'  PCA 2D: ID vs {display}...')
        ood_feat, _, _ = extract_features(model, all_ood_loaders[key], config.DEVICE)
        ood_np = ood_feat.numpy()
        n_ood_show = min(1000, len(ood_np))
        ood_sub = ood_np[rng.choice(len(ood_np), n_ood_show, replace=False)]

        # fit PCA on ID, then project OOD
        pca = PCA(n_components=2)
        id_proj_8 = pca.fit_transform(id_sub)
        ood_proj_8 = pca.transform(ood_sub)

        ax = axes_flat[i]
        ax.scatter(id_proj_8[:, 0], id_proj_8[:, 1],
                   c='steelblue', s=3, alpha=0.3, label='CIFAR-100 (ID)')
        ax.scatter(ood_proj_8[:, 0], ood_proj_8[:, 1],
                   c='crimson', s=3, alpha=0.3, label=display)
        ax.set_title(f'ID vs {display}', fontsize=11)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(fontsize=8, markerscale=3)

    # hide extra subplot
    axes_flat[-1].set_visible(False)

    fig.suptitle('Feature space: CIFAR-100 (ID) vs OOD datasets — PCA 2D', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_id_vs_ood.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_id_vs_ood.png'))
    plt.close(fig)

    # =========================================================================
    # 9. feature norm distribution: ID vs OOD
    # =========================================================================
    print('plotting feature norm distributions...')
    id_norms = np.linalg.norm(test_feat_np, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes_flat = axes.flatten()

    for i, (key, display) in enumerate(ood_datasets.items()):
        ood_feat, _, _ = extract_features(model, all_ood_loaders[key], config.DEVICE)
        ood_norms = np.linalg.norm(ood_feat.numpy(), axis=1)

        ax = axes_flat[i]
        ax.hist(id_norms, bins=60, alpha=0.6, color='steelblue', label='CIFAR-100 (ID)', density=True)
        ax.hist(ood_norms, bins=60, alpha=0.6, color='crimson', label=display, density=True)
        ax.set_title(f'ID vs {display}', fontsize=11)
        ax.set_xlabel(r'$\|h(x)\|_2$')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    axes_flat[-1].set_visible(False)
    fig.suptitle('Feature norm distributions: ID vs OOD', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'feature_norm_distributions.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'feature_norm_distributions.png'))
    plt.close(fig)

    # =========================================================================
    # 10. PCA 2D — ID class clusters (colored) + OOD in grey
    # =========================================================================
    print('plotting PCA 2D class clusters + OOD overlay...')

    # fit PCA on the 10-class subset, project OOD into same space
    pca_overlay = PCA(n_components=2)
    id_proj = pca_overlay.fit_transform(sub_feat)
    id_labs = sub_lab

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for i, (key, display) in enumerate(ood_datasets.items()):
        ood_feat, _, _ = extract_features(model, all_ood_loaders[key], config.DEVICE)
        ood_np = ood_feat.numpy()
        n_ood_show = min(1000, len(ood_np))
        ood_sub = ood_np[rng.choice(len(ood_np), n_ood_show, replace=False)]
        ood_proj = pca_overlay.transform(ood_sub)

        ax = axes_flat[i]
        # OOD in grey behind
        ax.scatter(ood_proj[:, 0], ood_proj[:, 1],
                   c='lightgray', s=4, alpha=0.4, label=f'{display} (OOD)', zorder=1)
        # ID colored by class on top
        ax.scatter(id_proj[:, 0], id_proj[:, 1],
                   c=id_labs, cmap='tab10', s=30, alpha=0.6, zorder=2)
        # class means for the 10 shown classes
        means_proj = pca_overlay.transform(class_means[show_classes])
        ax.scatter(means_proj[:, 0], means_proj[:, 1],
                   c='black', s=15, marker='x', linewidths=0.8, zorder=3)
        ax.set_title(f'ID clusters + {display}', fontsize=11)
        ax.set_xlabel(f'PC1 ({pca_overlay.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_overlay.explained_variance_ratio_[1]:.1%})')
        ax.legend(fontsize=8, markerscale=3, loc='upper right')

    axes_flat[-1].set_visible(False)

    fig.suptitle('CIFAR-100 class clusters (colored) + OOD (grey) — PCA 2D', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_clusters_with_ood.pdf'))
    fig.savefig(os.path.join(config.RESULTS_DIR, 'pca_2d_clusters_with_ood.png'))
    plt.close(fig)

    model.remove_hooks()
    print(f'all visualizations saved to {config.RESULTS_DIR}/')


if __name__ == '__main__':
    main()
