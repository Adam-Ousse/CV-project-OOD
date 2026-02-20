import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import logsumexp


def max_softmax_probability(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=1)
    return probs.max(dim=1)[0]


def max_logit_score(logits):
    return logits.max(dim=1)[0]


def energy_score(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)


class MahalanobisDetector:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_means = None
        self.precision = None

    def fit(self, features, labels):
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        self.class_means = np.array([
            features[labels == c].mean(axis=0) for c in range(self.num_classes)
        ])
        centered = features - self.class_means[labels]
        cov = EmpiricalCovariance(assume_centered=True).fit(centered)
        self.precision = cov.precision_

    def score(self, features):
        features = features.cpu().numpy().astype(np.float64)
        distances = []
        for c in range(self.num_classes):
            diff = features - self.class_means[c]
            distance = np.sum(diff @ self.precision * diff, axis=1)
            distances.append(distance)
        # higher score = more likely ID
        return -np.min(distances, axis=0)


class VimDetector:
    """ViM: Virtual logit Matching (Wang et al., CVPR 2022)."""
    def __init__(self, dim=300):
        self.dim = dim
        self.u = None       # training feature mean
        self.NS = None      # null space basis
        self.alpha = None   # calibration constant

    def fit(self, features, logits):
        """
        features: [N, p] train features
        logits: [N, C] train logits
        """
        features = features.cpu().numpy()
        logits = logits.cpu().numpy()

        self.u = features.mean(axis=0)

        # covariance of centered features
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)

        # eigh for symmetric matrix: guaranteed real, orthonormal eigenvectors, ascending order
        eig_vals, eig_vecs = np.linalg.eigh(ec.covariance_)

        # null space = eigenvectors with smallest eigenvalues (after DIM largest)
        # eigh returns ascending order, so top-dim are at the end
        self.NS = np.ascontiguousarray(eig_vecs[:, :-self.dim])

        # calibrate: alpha = mean(energy) / mean(vlogit) per VIM paper
        vlogit_train = np.linalg.norm(
            (features - self.u) @ self.NS, axis=-1
        )
        energy_train = logsumexp(logits, axis=-1)
        self.alpha = energy_train.mean() / vlogit_train.mean()

    def score(self, features, logits):
        features = features.cpu().numpy()
        logits = logits.cpu().numpy()

        vlogit = np.linalg.norm(
            (features - self.u) @ self.NS, axis=-1
        ) * self.alpha

        energy = logsumexp(logits, axis=-1)  # [N]
        # higher score = more likely ID
        return energy - vlogit


class NECODetector:
    """NECO: Neural Collapse inspired OOD detection (Ammar et al., 2024)."""
    def __init__(self, neco_dim=90, arch='resnet'):
        self.neco_dim = neco_dim
        self.arch = arch
        self.scaler = None
        self.pca = None

    def fit(self, features):
        features = features.cpu().numpy()
        # standardize features
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)
        # full PCA (all components)
        self.pca = PCA(n_components=features.shape[1])
        self.pca.fit(scaled)

    def score(self, features, logits):
        features = features.cpu().numpy()
        logits = logits.cpu().numpy()

        # transform through fitted scaler + PCA
        scaled = self.scaler.transform(features)
        projected_all = self.pca.transform(scaled)

        # take first neco_dim components (ETF subspace)
        projected_reduced = projected_all[:, :self.neco_dim]

        # ratio = norm in ETF subspace / norm in full (scaled) space
        norm_reduced = np.linalg.norm(projected_reduced, axis=1)
        norm_full = np.linalg.norm(scaled, axis=1) + 1e-8
        ratio = norm_reduced / norm_full

        # neco score = ratio * energy per original paper
        energy = logsumexp(logits, axis=-1)
        return ratio * energy
