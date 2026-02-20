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
        return -np.min(distances, axis=0)


class VimDetector:
    def __init__(self, dim=300):
        self.dim = dim
        self.u = None
        self.NS = None
        self.alpha = None

    def fit(self, features, logits):
        features = features.cpu().numpy()
        logits = logits.cpu().numpy()

        self.u = features.mean(axis=0)

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)

        # eigh: symmetric matrix, ascending eigenvalue order
        eig_vals, eig_vecs = np.linalg.eigh(ec.covariance_)

        # null space: eigenvecs with smallest eigenvalues
        self.NS = np.ascontiguousarray(eig_vecs[:, :-self.dim])

        # calibrate alpha
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

        energy = logsumexp(logits, axis=-1)
        return energy - vlogit


class NECODetector:
    def __init__(self, neco_dim=90, arch='resnet'):
        self.neco_dim = neco_dim
        self.arch = arch
        self.scaler = None
        self.pca = None

    def fit(self, features):
        features = features.cpu().numpy()
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)
        self.pca = PCA(n_components=features.shape[1])
        self.pca.fit(scaled)

    def score(self, features, logits):
        features = features.cpu().numpy()
        logits = logits.cpu().numpy()

        scaled = self.scaler.transform(features)
        projected_all = self.pca.transform(scaled)

        # etf subspace
        projected_reduced = projected_all[:, :self.neco_dim]

        norm_reduced = np.linalg.norm(projected_reduced, axis=1)
        norm_full = np.linalg.norm(scaled, axis=1) + 1e-8
        ratio = norm_reduced / norm_full

        energy = logsumexp(logits, axis=-1)
        return ratio * energy
