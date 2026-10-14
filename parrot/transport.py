"""Optimal Transport Computation for PARROT"""

import torch
import torch.nn.functional as F
import numpy as np
import ot as pot


def optimal_transport_distance(
    p_features: torch.Tensor,
    q_features: torch.Tensor,
    method: str = "wasserstein",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Optimal Transport distance between two feature distributions.

    Supports:
    - Wasserstein distance (Optimal Transport)
    - Sliced Wasserstein distance (faster approximation)

    Args:
        p_features: Features from first model [batch_size, feature_dim]
        q_features: Features from second model [batch_size, feature_dim]
        method: "wasserstein" or "sliced_wasserstein"
        eps: Numerical epsilon for stability

    Returns:
        Optimal Transport distance
    """
    batch_size = p_features.size(0)

    if p_features.is_cuda:
        p_features = p_features.cpu()
    if q_features.is_cuda:
        q_features = q_features.cpu()

    p_np = p_features.detach().numpy()
    q_np = q_features.detach().numpy()

    if method == "wasserstein":
        # Full Wasserstein distance
        distances = []
        for i in range(batch_size):
            # Compute cost matrix
            p = p_np[i]  # [feature_dim]
            q = q_np[i]  # [feature_dim]

            # Create uniform weights
            a = np.ones(len(p)) / len(p)
            b = np.ones(len(q)) / len(q)

            # Compute Euclidean cost matrix
            M = pot.utils.dist(p.reshape(-1, 1), q.reshape(-1, 1))

            # Solve OT problem
            ot_dist = pot.emd2(a, b, M)
            distances.append(ot_dist)

        distance = torch.tensor(distances, dtype=p_features.dtype)

    elif method == "sliced_wasserstein":
        # Faster Sliced Wasserstein distance
        distances = []
        for i in range(batch_size):
            p = p_np[i]  # [feature_dim]
            q = q_np[i]  # [feature_dim]

            # Compute sliced Wasserstein distance
            sw_dist = pot.sliced_wasserstein_distance(
                p.reshape(1, -1),
                q.reshape(1, -1),
                n_projections=50
            )
            distances.append(sw_dist)

        distance = torch.tensor(distances, dtype=p_features.dtype)

    else:
        raise ValueError(f"Unknown method: {method}")

    return distance.clamp(min=0)


def compute_transport_plan(
    p_features: torch.Tensor,
    q_features: torch.Tensor
) -> torch.Tensor:
    """
    Compute the optimal transport plan (coupling) between distributions.

    Returns the coupling matrix that maps probability mass from p to q.

    Args:
        p_features: Features from first model
        q_features: Features from second model

    Returns:
        Transport plan (coupling matrix)
    """
    p_np = p_features.cpu().detach().numpy()
    q_np = q_features.cpu().detach().numpy()

    # Compute cost matrix (Euclidean distance)
    M = pot.utils.dist(p_np, q_np)

    # Uniform weights
    a = np.ones(len(p_np)) / len(p_np)
    b = np.ones(len(q_np)) / len(q_np)

    # Compute optimal transport plan
    P = pot.emd(a, b, M)

    return torch.from_numpy(P).float()
