import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import config
from src.model.model import ResNet18Classifier
from src.data.data_loader import get_cifar100_loaders

sns.set_theme(style='whitegrid', font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

csv_path = os.path.join(config.RESULTS_DIR, 'nc_metrics.csv')
df = pd.read_csv(csv_path)


def load_model():
    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
                      map_location=config.DEVICE)
    model = ResNet18Classifier(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def extract_all_features(model, loader):
    model.eval()
    feats, labs = [], []
    layer_accum = {l: [] for l in ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            _, f, layer_feats = model(x, return_features=True)
            feats.append(f.cpu())
            labs.append(y)
            for name in layer_accum:
                if name in layer_feats:
                    layer_accum[name].append(layer_feats[name].cpu())
    features = torch.cat(feats)
    labels   = torch.cat(labs)
    layer_features = {k: torch.cat(v) for k, v in layer_accum.items() if v}
    return features, labels, layer_features


def compute_class_geometry(features, labels, classifier):
    C = config.NUM_CLASSES
    features = features.to(config.DEVICE)
    labels   = labels.to(config.DEVICE)

    global_mean = features.mean(0)
    means = torch.stack([features[labels == c].mean(0) for c in range(C)])  # [C, p]

    per_class_var = torch.zeros(C)
    for c in range(C):
        fc = features[labels == c]
        per_class_var[c] = ((fc - means[c]) ** 2).sum(1).mean()

    W = classifier.weight.data  # [C, p]
    M = (means - global_mean)   # [C, p]  centered
    cos = torch.nn.functional.cosine_similarity(W, M, dim=1)  # [C]

    return means.cpu(), global_mean.cpu(), per_class_var.cpu(), cos.cpu()


def nc1_per_layer(layer_features, labels):
    labels = labels.to(config.DEVICE)
    results = {}
    for name, feats in layer_features.items():
        feats = feats.to(config.DEVICE)
        C = config.NUM_CLASSES
        g = feats.mean(0)
        means = torch.stack([feats[labels == c].mean(0) for c in range(C)])
        centered_means = means - g
        Sb = (centered_means.T @ centered_means) / C
        Sw = torch.zeros(feats.shape[1], feats.shape[1], device=config.DEVICE)
        for c in range(C):
            fc = feats[labels == c]
            d  = fc - means[c]
            Sw += (d.T @ d) / fc.shape[0]
        Sw /= C
        tr_w = torch.trace(Sw).item()
        tr_b = torch.trace(Sb).item()
        results[name] = tr_w / tr_b if tr_b > 1e-10 else float('inf')
    return results

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

nc_configs = [
    ('nc1', r'NC1 — $\mathrm{Tr}(\Sigma_W) \,/\, \mathrm{Tr}(\Sigma_B)$', 'tab:blue', r'$\downarrow 0$'),
    ('nc2', r'NC2 — ETF deviation', 'tab:orange', r'$\downarrow 0$'),
    ('nc3', r'NC3 — Self-duality ($1 - \cos$)', 'tab:green', r'$\downarrow 0$'),
    ('nc4', r'NC4 — NCC agreement', 'tab:red', r'$\uparrow 1$'),
]

for ax, (col, label, color, target) in zip(axes, nc_configs):
    sns.lineplot(x='epoch', y=col, data=df, ax=ax, color=color, linewidth=1.2)
    for ms in config.LR_MILESTONES:
        ax.axvline(x=ms, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_ylabel(label, fontsize=13)
    ax.set_xlabel('')
    ax.annotate(f'target: {target}', xy=(0.98, 0.92), xycoords='axes fraction',
                ha='right', va='top', fontsize=11, color='gray')

axes[-1].set_xlabel('Epoch', fontsize=14)
fig.align_ylabels(axes)
fig.suptitle(r'Neural Collapse Metrics (ResNet-18 / CIFAR-100)', fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(config.RESULTS_DIR, 'nc_metrics.pdf'))
fig.savefig(os.path.join(config.RESULTS_DIR, 'nc_metrics.png'))
plt.close(fig)

fig2, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

sns.lineplot(x='epoch', y='train_loss', data=df, ax=ax_loss, label='Train', linewidth=1.2)
sns.lineplot(x='epoch', y='val_loss', data=df, ax=ax_loss, label='Val', linewidth=1.2)
for ms in config.LR_MILESTONES:
    ax_loss.axvline(x=ms, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
ax_loss.set_ylabel('Loss', fontsize=14)
ax_loss.set_xlabel('')
ax_loss.legend(fontsize=12)

sns.lineplot(x='epoch', y='train_acc', data=df, ax=ax_acc, label='Train', linewidth=1.2)
sns.lineplot(x='epoch', y='val_acc', data=df, ax=ax_acc, label='Val', linewidth=1.2)
for ms in config.LR_MILESTONES:
    ax_acc.axvline(x=ms, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
ax_acc.set_ylabel(r'Accuracy (\%)', fontsize=14)
ax_acc.set_xlabel('Epoch', fontsize=14)
ax_acc.legend(fontsize=12)

fig2.suptitle(r'Training Curves (ResNet-18 / CIFAR-100)', fontsize=16, y=1.01)
fig2.tight_layout()
fig2.savefig(os.path.join(config.RESULTS_DIR, 'training_curves.pdf'))
fig2.savefig(os.path.join(config.RESULTS_DIR, 'training_curves.png'))
plt.close(fig2)

print(f'saved to {config.RESULTS_DIR}/nc_metrics.pdf and training_curves.pdf')


# nc5 cross-layer cka
fig5, (ax_cka, ax_corr) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

sns.lineplot(x='epoch', y='nc5_cka',         data=df, ax=ax_cka,  color='tab:purple', linewidth=1.2)
sns.lineplot(x='epoch', y='nc5_correlation', data=df, ax=ax_corr, color='tab:brown',  linewidth=1.2)

for ms in config.LR_MILESTONES:
    ax_cka.axvline(x=ms,  color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax_corr.axvline(x=ms, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

ax_cka.set_ylabel('NC5 — CKA (↑ 1)', fontsize=13)
ax_corr.set_ylabel('NC5 — Gram correlation (↓ 0)', fontsize=13)
ax_corr.set_xlabel('Epoch', fontsize=14)
fig5.suptitle('NC5: Cross-layer subspace alignment (ResNet-18 / CIFAR-100)', fontsize=15, y=1.01)
fig5.tight_layout()
fig5.savefig(os.path.join(config.RESULTS_DIR, 'nc5_cross_layer.pdf'))
fig5.savefig(os.path.join(config.RESULTS_DIR, 'nc5_cross_layer.png'))
plt.close(fig5)
print('saved nc5_cross_layer.pdf')


print('loading model and computing features for geometry plots...')
model   = load_model()
_, _, test_loader, clean_loader = get_cifar100_loaders()
features, labels, layer_feats = extract_all_features(model, clean_loader)
means, global_mean, per_class_var, per_class_cos = compute_class_geometry(
    features, labels, model.model.fc)


# class mean distance matrix
M_c = (means - global_mean)                          # [C, p] centered
dists = torch.cdist(M_c, M_c, p=2).numpy()          # [C, C]

fig_d, ax_d = plt.subplots(figsize=(9, 7))
im = ax_d.imshow(dists, aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax_d, label='L2 distance')
ax_d.set_title('Pairwise L2 distances between class means\n(centered, final model)', fontsize=13)
ax_d.set_xlabel('class index')
ax_d.set_ylabel('class index')
fig_d.tight_layout()
fig_d.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_distances.pdf'))
fig_d.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_distances.png'))
plt.close(fig_d)
print('saved class_mean_distances.pdf')

off_diag = dists[np.triu_indices(config.NUM_CLASSES, k=1)]
fig_dh, ax_dh = plt.subplots(figsize=(7, 4))
ax_dh.hist(off_diag, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
ax_dh.set_xlabel('L2 distance between class means')
ax_dh.set_ylabel('count')
ax_dh.set_title(f'Distribution of inter-class mean distances  (μ={off_diag.mean():.2f}, σ={off_diag.std():.2f})')
fig_dh.tight_layout()
fig_dh.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_dist_hist.pdf'))
fig_dh.savefig(os.path.join(config.RESULTS_DIR, 'class_mean_dist_hist.png'))
plt.close(fig_dh)


# within-class variance
var_np  = per_class_var.numpy()
order   = np.argsort(var_np)[::-1]

fig_v, ax_v = plt.subplots(figsize=(14, 4))
ax_v.bar(np.arange(config.NUM_CLASSES), var_np[order], color='steelblue', width=1.0)
ax_v.axhline(var_np.mean(), color='tomato', linestyle='--', linewidth=1.2,
             label=f'mean = {var_np.mean():.2f}')
ax_v.set_xlabel('class (sorted by variance)')
ax_v.set_ylabel('within-class variance')
ax_v.set_title('Per-class within-class variance (final model)')
ax_v.legend()
fig_v.tight_layout()
fig_v.savefig(os.path.join(config.RESULTS_DIR, 'within_class_variance.pdf'))
fig_v.savefig(os.path.join(config.RESULTS_DIR, 'within_class_variance.png'))
plt.close(fig_v)
print('saved within_class_variance.pdf')


# nc3 cosine similarity
cos_np = per_class_cos.numpy()
order_cos = np.argsort(cos_np)

colors = ['tomato' if c < 0.9 else 'steelblue' for c in cos_np[order_cos]]
fig_c, ax_c = plt.subplots(figsize=(14, 4))
ax_c.bar(np.arange(config.NUM_CLASSES), cos_np[order_cos], color=colors, width=1.0)
ax_c.axhline(cos_np.mean(), color='black', linestyle='--', linewidth=1.2,
             label=f'mean = {cos_np.mean():.3f}')
ax_c.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax_c.set_xlabel('class (sorted by cosine similarity)')
ax_c.set_ylabel(r'$\cos(w_c,\, \mu_c - \mu_G)$')
ax_c.set_title('Per-class cosine similarity: classifier weight vs centered class mean  (NC3)')
ax_c.legend()
fig_c.tight_layout()
fig_c.savefig(os.path.join(config.RESULTS_DIR, 'cosine_sim_weights_means.pdf'))
fig_c.savefig(os.path.join(config.RESULTS_DIR, 'cosine_sim_weights_means.png'))
plt.close(fig_c)
print('saved cosine_sim_weights_means.pdf')


# nc1 across layers
layer_nc1 = nc1_per_layer(layer_feats, labels)

layer_names  = list(layer_nc1.keys())
layer_values = [layer_nc1[k] for k in layer_names]

fig_l, ax_l = plt.subplots(figsize=(8, 4))
bars = ax_l.bar(layer_names, layer_values, color='mediumseagreen', edgecolor='white')
ax_l.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
ax_l.set_ylabel(r'NC1  $\mathrm{Tr}(\Sigma_W)\,/\,\mathrm{Tr}(\Sigma_B)$')
ax_l.set_title('Neural Collapse (NC1) across ResNet-18 layers — final model')
ax_l.set_ylim(0, max(layer_values) * 1.18)
fig_l.tight_layout()
fig_l.savefig(os.path.join(config.RESULTS_DIR, 'nc1_across_layers.pdf'))
fig_l.savefig(os.path.join(config.RESULTS_DIR, 'nc1_across_layers.png'))
plt.close(fig_l)
print('saved nc1_across_layers.pdf')

model.remove_hooks()
print('done.')

