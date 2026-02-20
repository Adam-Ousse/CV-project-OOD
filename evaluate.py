import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import config
from src.model.model import ResNet18Classifier
from src.data.data_loader import get_cifar100_loaders, get_ood_loader
from src.ood.ood_scores import (max_softmax_probability, max_logit_score, energy_score,
                                MahalanobisDetector, VimDetector, NECODetector)
from src.nc.nc import NCMetrics
from sklearn.metrics import roc_auc_score, roc_curve


def extract_features(model, loader, device):
    """extract avgpool features, labels, and logits from a loader."""
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


def compute_fpr_at_tpr(y_true, scores, tpr_threshold=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.argmin(np.abs(tpr - tpr_threshold))
    return fpr[idx]


def evaluate_single_ood(id_features, id_labels, id_logits,
                        ood_features, ood_logits,
                        train_features, train_labels, train_logits,
                        maha, vim, neco):
    """evaluate all methods for one OOD dataset. returns (metrics_dict, scores_dict)."""
    n_id = len(id_features)
    n_ood = len(ood_features)
    y_true = np.concatenate([np.ones(n_id), np.zeros(n_ood)])
    results = {}
    raw_scores = {}  # {method: {'id': arr, 'ood': arr}}

    def _record(name, id_s, ood_s):
        raw_scores[name] = {'id': id_s, 'ood': ood_s}
        scores = np.concatenate([id_s, ood_s])
        results[name] = {'auroc': roc_auc_score(y_true, scores),
                         'fpr95': compute_fpr_at_tpr(y_true, scores)}

    _record('MSP',
            max_softmax_probability(id_logits, config.TEMPERATURE).numpy(),
            max_softmax_probability(ood_logits, config.TEMPERATURE).numpy())

    _record('MaxLogit',
            max_logit_score(id_logits).numpy(),
            max_logit_score(ood_logits).numpy())

    _record('Energy',
            energy_score(id_logits, config.TEMPERATURE).numpy(),
            energy_score(ood_logits, config.TEMPERATURE).numpy())

    _record('Mahalanobis', maha.score(id_features), maha.score(ood_features))

    _record('ViM',
            vim.score(id_features, id_logits),
            vim.score(ood_features, ood_logits))

    _record('NECO',
            neco.score(id_features, id_logits),
            neco.score(ood_features, ood_logits))

    return results, raw_scores


def plot_score_distributions(scores_near, scores_far, ood_near_name, ood_far_name, save_dir):
    """2 figures (near / far), 6 subplots each — KDE of ID vs OOD scores per method."""
    methods = list(scores_near.keys())
    n = len(methods)

    for tag, scores, ood_name in [('near', scores_near, ood_near_name),
                                   ('far',  scores_far,  ood_far_name)]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'OOD score distributions — ID: CIFAR-100 vs {ood_name}', fontsize=13)

        for ax, method in zip(axes.flat, methods):
            id_s  = scores[method]['id']
            ood_s = scores[method]['ood']

            # clip extreme outliers for visual clarity (1st–99th percentile)
            lo = min(np.percentile(id_s, 0.5), np.percentile(ood_s, 0.5))
            hi = max(np.percentile(id_s, 99.5), np.percentile(ood_s, 99.5))
            bins = np.linspace(lo, hi, 60)

            ax.hist(id_s,  bins=bins, density=True, alpha=0.55, color='steelblue', label='ID (CIFAR-100)')
            ax.hist(ood_s, bins=bins, density=True, alpha=0.55, color='tomato',    label=f'OOD ({ood_name})')
            ax.set_title(method, fontsize=11)
            ax.set_xlabel('score')
            ax.set_ylabel('density')
            ax.legend(fontsize=8)

        fig.tight_layout()
        path = os.path.join(save_dir, f'score_dist_{tag}_ood.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'saved {path}')


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # load model
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'),
                            map_location=config.DEVICE)
    model = ResNet18Classifier(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ID data
    train_loader, val_loader, test_loader, clean_loader = get_cifar100_loaders()
    print('extracting ID train features...')
    train_features, train_labels, train_logits = extract_features(model, clean_loader, config.DEVICE)
    print('extracting ID test features...')
    test_features, test_labels, test_logits = extract_features(model, test_loader, config.DEVICE)
    n_id_test = len(test_features)

    # fit detectors on train set
    print('fitting mahalanobis...')
    maha = MahalanobisDetector(config.NUM_CLASSES)
    maha.fit(train_features, train_labels)

    print('fitting vim...')
    # null space dim = num_classes (100); principal subspace = feature_dim - num_classes
    feature_dim = train_features.shape[1]  # 512 for resnet-18
    vim = VimDetector(dim=feature_dim - config.NUM_CLASSES)
    vim.fit(train_features, train_logits)

    print('fitting neco...')
    neco = NECODetector(neco_dim=90, arch='resnet')
    neco.fit(train_features)

    # OOD datasets
    ood_config = {
        'near': {'cifar10': 'CIFAR-10', 'tiny_imagenet': 'Tiny ImageNet'},
        'far':  {'mnist': 'MNIST', 'svhn': 'SVHN', 'textures': 'Textures'},
    }

    all_rows = []
    scores_near, scores_far = {}, {}
    near_name, far_name = 'Tiny ImageNet', 'SVHN'

    for group, datasets in ood_config.items():
        ood_loaders = get_ood_loader(group)
        for key, display_name in datasets.items():
            print(f'\nevaluating {display_name} ({group}-OOD)...')
            loader = ood_loaders[key]
            ood_features, ood_labels, ood_logits = extract_features(model, loader, config.DEVICE)

            # subsample ID test to match OOD size (for textures etc.)
            n_ood = len(ood_features)
            if n_ood < n_id_test:
                idx = torch.randperm(n_id_test)[:n_ood]
                id_feat = test_features[idx]
                id_lab = test_labels[idx]
                id_log = test_logits[idx]
            else:
                id_feat = test_features
                id_lab = test_labels
                id_log = test_logits

            results, raw_scores = evaluate_single_ood(
                id_feat, id_lab, id_log,
                ood_features, ood_logits,
                train_features, train_labels, train_logits,
                maha, vim, neco,
            )

            # store scores for distribution plots
            if key == 'tiny_imagenet':
                scores_near = raw_scores
                near_name = display_name
            elif key == 'svhn':
                scores_far = raw_scores
                far_name = display_name

            for method, metrics in results.items():
                all_rows.append({
                    'group': group,
                    'ood_dataset': display_name,
                    'method': method,
                    'AUROC': metrics['auroc'],
                    'FPR@95': metrics['fpr95'],
                })

    # build dataframes
    df = pd.DataFrame(all_rows)

    # score distribution plots: CIFAR-100 vs Tiny ImageNet (near) and SVHN (far)
    plot_score_distributions(scores_near, scores_far, near_name, far_name, config.RESULTS_DIR)

    # per-dataset table
    pivot_auroc = df.pivot_table(index='method', columns='ood_dataset', values='AUROC')
    pivot_fpr = df.pivot_table(index='method', columns='ood_dataset', values='FPR@95')

    # near / far averages
    near_datasets = list(ood_config['near'].values())
    far_datasets = list(ood_config['far'].values())

    summary_rows = []
    for method in df['method'].unique():
        m_df = df[df['method'] == method]
        near_auroc = m_df[m_df['group'] == 'near']['AUROC'].mean()
        near_fpr = m_df[m_df['group'] == 'near']['FPR@95'].mean()
        far_auroc = m_df[m_df['group'] == 'far']['AUROC'].mean()
        far_fpr = m_df[m_df['group'] == 'far']['FPR@95'].mean()
        summary_rows.append({
            'method': method,
            'Near-OOD AUROC': near_auroc, 'Near-OOD FPR@95': near_fpr,
            'Far-OOD AUROC': far_auroc, 'Far-OOD FPR@95': far_fpr,
        })
    df_summary = pd.DataFrame(summary_rows).set_index('method')

    # print tables
    print('\n' + '=' * 80)
    print('AUROC per dataset')
    print('=' * 80)
    print(pivot_auroc.to_string(float_format='%.4f'))

    print('\n' + '=' * 80)
    print('FPR@95 per dataset')
    print('=' * 80)
    print(pivot_fpr.to_string(float_format='%.4f'))

    print('\n' + '=' * 80)
    print('Summary (Near / Far OOD averages)')
    print('=' * 80)
    print(df_summary.to_string(float_format='%.4f'))

    # NC metrics on train set (last epoch snapshot)
    print('\ncomputing NC metrics on train features...')
    nc = NCMetrics(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    nc_metrics, means, global_mean = nc.compute_all_metrics(
        train_features.to(config.DEVICE),
        train_labels.to(config.DEVICE),
        model.model.fc,
    )
    nc_df = pd.DataFrame([{
        'NC1 (Tr(Σ_W)/Tr(Σ_B))': nc_metrics['nc1'],
        'NC2 (ETF deviation)': nc_metrics['nc2'],
        'NC2 cos MSE': nc_metrics['nc2_cos'],
        'NC3 (1 - cos)': nc_metrics['nc3'],
        'NC3 dist': nc_metrics['nc3_dist'],
        'NC4 (NCC agreement)': nc_metrics['nc4'],
    }])

    print('\n' + '=' * 80)
    print('Neural Collapse Metrics (final model)')
    print('=' * 80)
    print(nc_df.to_string(index=False, float_format='%.6f'))

    # save all to csv
    df.to_csv(os.path.join(config.RESULTS_DIR, 'ood_results_full.csv'), index=False)
    df_summary.to_csv(os.path.join(config.RESULTS_DIR, 'ood_results_summary.csv'))
    nc_df.to_csv(os.path.join(config.RESULTS_DIR, 'nc_metrics_final.csv'), index=False)
    print(f'\nresults saved to {config.RESULTS_DIR}/')

    model.remove_hooks()


if __name__ == '__main__':
    main()
