import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Simplified UMAP implementation per sheet02, exercise 1.
# Uses kNN graph for attraction and random pairs for repulsion.


def load_jet_data(base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load jet features and labels from sheet01 data."""
    features_path = base_dir / "sheet01" / "data" / "dijet_features.npy"
    labels_path = base_dir / "sheet01" / "data" / "dijet_labels.npy"
    X = np.load(features_path)  # shape (p, N)
    y = np.load(labels_path)
    return X.astype(np.float64), y


def knn_graph(X: np.ndarray, k: int = 15) -> np.ndarray:
    """Return symmetrized kNN edge list as array of shape (M, 2) with i<j."""
    # X is shape (p, N)
    _, N = X.shape
    # pairwise distances (N, N)
    dists = np.linalg.norm(X[:, None, :] - X[:, :, None], axis=0)
    knn = np.argpartition(dists, kth=k+1, axis=1)[:, 1:k+1]  # skip self (0th)
    edges = set()
    for i in range(N):
        for j in knn[i]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def sample_repulsive_pairs(N: int, m: int) -> np.ndarray:
    """Sample m random pairs (i,j) with i<j uniformly."""
    rng = np.random.default_rng()
    pairs = set()
    while len(pairs) < m:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        pairs.add((a, b))
    return np.array(list(pairs), dtype=np.int64)


def phi_attr(d2: np.ndarray) -> np.ndarray:
    return np.log1p(d2)


def phi_rep(d2: np.ndarray, c: float) -> np.ndarray:
    return c * (1.0 / (1.0 + d2))


def grad_phi_attr(d: np.ndarray, d2: np.ndarray) -> np.ndarray:
    # d shape (..., 2), d2 shape (...,)
    coeff = 2.0 / (1.0 + d2)[..., None]
    return coeff * d


def grad_phi_rep(d: np.ndarray, d2: np.ndarray, c: float) -> np.ndarray:
    coeff = -2.0 * c / (1.0 + d2)[..., None] ** 2
    return coeff * d


def force_layout(edges: np.ndarray,
                  X_low: np.ndarray,
                  steps: int = 400,
                  lr: float = 0.01,
                  min_lr: float = 1e-4,
                  c: float = 10.0,
                  rep_mul: int = 5,
                  rng_seed: int = 42) -> np.ndarray:
    """Force-directed layout with attractive edges and random repulsive pairs resampled each step."""
    rng = np.random.default_rng(rng_seed)
    N = X_low.shape[1]
    for t in range(steps):
        # resample repulsive pairs
        rep_pairs = sample_repulsive_pairs(N, rep_mul * N)

        grad = np.zeros_like(X_low)  # shape (2, N)

        # Attractive forces
        d_attr = X_low[:, edges[:, 0]] - X_low[:, edges[:, 1]]  # shape (2, M)
        d2_attr = np.sum(d_attr ** 2, axis=0)
        g_attr = grad_phi_attr(d_attr.T, d2_attr).T  # shape (2, M)
        np.add.at(grad, (slice(None), edges[:, 0]), g_attr)
        np.add.at(grad, (slice(None), edges[:, 1]), -g_attr)

        # Repulsive forces
        d_rep = X_low[:, rep_pairs[:, 0]] - X_low[:, rep_pairs[:, 1]]
        d2_rep = np.sum(d_rep ** 2, axis=0)
        g_rep = grad_phi_rep(d_rep.T, d2_rep, c=c).T
        np.add.at(grad, (slice(None), rep_pairs[:, 0]), g_rep)
        np.add.at(grad, (slice(None), rep_pairs[:, 1]), -g_rep)

        # learning rate schedule (linear decay to min_lr)
        lr_t = lr * (1.0 - t / steps) + min_lr
        X_low -= lr_t * grad

    return X_low


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Return 2D PCA projection of X (p, N)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    # SVD on covariance
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    comps = U[:, :2].T  # (2, p)
    return comps @ Xc


def run_simplified_umap(k: int = 15,
                        c: float = 10.0,
                        steps: int = 400,
                        lr: float = 0.01,
                        min_lr: float = 1e-4,
                        seed: int = 42) -> None:
    base = Path(__file__).resolve().parents[1]
    X_high, labels = load_jet_data(base)
    p, N = X_high.shape
    print(f"Loaded jets: X shape={X_high.shape}, labels shape={labels.shape}")

    edges = knn_graph(X_high, k=k)
    print(f"kNN edges: {len(edges)}")

    rng = np.random.default_rng(seed)
    X_low0 = rng.normal(scale=0.01, size=(2, N))

    X_low = force_layout(edges, X_low0, steps=steps, lr=lr, min_lr=min_lr, c=c, rep_mul=5, rng_seed=seed)

    # PCA baseline
    X_pca = pca_2d(X_high)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = {0: "red", 1: "green", 2: "blue"}

    axes[0].scatter(X_low[0], X_low[1], c=[cmap[int(l)] for l in labels], s=8, alpha=0.7)
    axes[0].set_title("Simplified UMAP (2D)")
    axes[0].set_xlabel("dim1")
    axes[0].set_ylabel("dim2")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(X_pca[0], X_pca[1], c=[cmap[int(l)] for l in labels], s=8, alpha=0.7)
    axes[1].set_title("PCA (2D)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simplified_umap()
