import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class NCMetrics:
    def __init__(self, num_classes, device='cuda'):
        self.C = num_classes
        self.device = device
        self.history = defaultdict(list)
        
    def compute_all_metrics(self, features, labels, classifier):
        metrics = {}
        means, global_mean, within_cov, between_cov = self._compute_moments(features, labels)
        metrics['nc1'] = self._compute_nc1(within_cov, between_cov)
        metrics['nc2'], metrics['nc2_cos'] = self._compute_nc2(means, global_mean)
        metrics['nc3'], metrics['nc3_dist'] = self._compute_nc3(means, global_mean, classifier)
        metrics['nc4'] = self._compute_nc4(features, labels, means, classifier)
        
        for key, value in metrics.items():
            self.history[key].append(value)
            
        return metrics, means, global_mean
    
    def compute_nc5_cross_layer(self, layer_means_dict):
        # cross-layer gram correlation and linear CKA
        if len(layer_means_dict) < 2:
            return {'nc5_correlation': 0.0, 'nc5_cka': 0.0}
        
        layer_names = list(layer_means_dict.keys())
        n_layers = len(layer_names)
        
        # normalize each layer's class means
        normalized_means = {}
        for name, means in layer_means_dict.items():
            norms = torch.norm(means, dim=0, keepdim=True)
            normalized_means[name] = means / (norms + 1e-10)
        
        # pairwise gram correlations
        correlations = []
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                m1 = normalized_means[layer_names[i]]
                m2 = normalized_means[layer_names[j]]
                
                gram1 = m1.T @ m1
                gram2 = m2.T @ m2
                
                g1_norm = torch.norm(gram1, p='fro')
                g2_norm = torch.norm(gram2, p='fro')
                gram1_n = gram1 / (g1_norm + 1e-10)
                gram2_n = gram2 / (g2_norm + 1e-10)
                
                correlation = (gram1_n * gram2_n).sum().item()
                correlations.append(abs(correlation))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # linear CKA
        cka_values = []
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                m1 = layer_means_dict[layer_names[i]]
                m2 = layer_means_dict[layer_names[j]]
                
                k1 = m1.T @ m1
                k2 = m2.T @ m2
                
                hsic_12 = (k1 * k2).sum().item()
                hsic_11 = (k1 * k1).sum().item()
                hsic_22 = (k2 * k2).sum().item()
                
                denom = np.sqrt(hsic_11 * hsic_22)
                cka = hsic_12 / denom if denom > 1e-10 else 0.0
                cka_values.append(cka)
        
        avg_cka = np.mean(cka_values) if cka_values else 0.0
        
        return {
            'nc5_correlation': avg_correlation,
            'nc5_cka': avg_cka
        }
    
    def _compute_moments(self, features, labels):
        N, p = features.shape
        C = self.C
        
        global_mean = features.mean(dim=0)          # [p]
        
        means = torch.zeros(p, C, device=self.device)
        counts = torch.zeros(C, device=self.device)
        
        for c in range(C):
            mask = (labels == c)
            count = mask.sum().item()
            if count > 0:
                means[:, c] = features[mask].mean(dim=0)
                counts[c] = count
        
        centered_means = means - global_mean.unsqueeze(1)  # [p, C]
        between_cov = (centered_means @ centered_means.T) / C  # [p, p]
        
        within_cov = torch.zeros(p, p, device=self.device)
        valid_classes = 0
        for c in range(C):
            mask = (labels == c)
            n_c = mask.sum().item()
            if n_c > 0:
                centered = features[mask] - means[:, c].unsqueeze(0)
                within_cov += (centered.T @ centered) / n_c
                valid_classes += 1
        
        if valid_classes > 0:
            within_cov /= valid_classes
        
        return means, global_mean, within_cov, between_cov
    
    def _compute_nc1(self, within_cov, between_cov):
        # Tr(Σ_W) / Tr(Σ_B)
        trace_w = torch.trace(within_cov).item()
        trace_b = torch.trace(between_cov).item()
        return trace_w / trace_b if trace_b > 1e-10 else float('inf')
    
    def _compute_nc2(self, means, global_mean):
        M_centered = means - global_mean.unsqueeze(1)  # [p, C]
        norms = torch.norm(M_centered, dim=0)           # [C]
        
        if norms.mean() < 1e-10:
            return float('inf'), float('inf')
        
        M_normalized = M_centered / (norms.unsqueeze(0) + 1e-10)  # [p, C]
        G = M_normalized.T @ M_normalized  # [C, C]
        
        G_ideal = torch.full((self.C, self.C), -1.0 / (self.C - 1), device=self.device)
        G_ideal.fill_diagonal_(1.0)
        
        nc2 = torch.norm(G - G_ideal, p='fro').item() / self.C
        
        off_diag_mask = ~torch.eye(self.C, dtype=torch.bool, device=self.device)
        off_diag_vals = G[off_diag_mask]
        target = -1.0 / (self.C - 1)
        nc2_cos = ((off_diag_vals - target) ** 2).mean().item()
        
        return nc2, nc2_cos
    
    def _compute_nc3(self, means, global_mean, classifier):
        W = classifier.weight.data
        M_dot = (means - global_mean.unsqueeze(1)).T
        
        w_norms = torch.norm(W, dim=1, keepdim=True)
        m_norms = torch.norm(M_dot, dim=1, keepdim=True)
        
        valid = (w_norms.squeeze() > 1e-10) & (m_norms.squeeze() > 1e-10)
        
        if valid.sum() == 0:
            return 1.0, float('inf')
        
        W_n = W[valid] / w_norms[valid]
        M_n = M_dot[valid] / m_norms[valid]
        
        cosines = (W_n * M_n).sum(dim=1)
        nc3 = (1.0 - cosines).mean().item()  # 0 = perfect alignment
        
        # frobenius distance after per-class normalisation
        nc3_dist = torch.norm(W_n - M_n, p='fro').item() ** 2 / valid.sum().item()
        
        return nc3, nc3_dist
    
    def _compute_nc4(self, features, labels, means, classifier):
        with torch.no_grad():
            logits = classifier(features)
            net_preds = logits.argmax(dim=1)
        
        distances = torch.cdist(features, means.T, p=2) ** 2
        ncc_preds = distances.argmin(dim=1)
        
        agreement = (net_preds == ncc_preds).float().mean().item()
        return agreement
    
    def get_history(self):
        return dict(self.history)

