import numpy as np

from sfm_experiments.metrics import compute_ate


def pose_from_center(center):
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = -np.asarray(center)
    return qvec, tvec


def test_compute_ate_is_similarity_invariant():
    est_centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    ]
    estimated = {f"img{i}": pose_from_center(c) for i, c in enumerate(est_centers)}

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 2.5
    translation = np.array([5.0, -3.0, 0.5])

    gt_centers = [scale * rotation @ c + translation for c in est_centers]
    ground_truth = {f"img{i}": pose_from_center(c) for i, c in enumerate(gt_centers)}

    ate = compute_ate(estimated, ground_truth)

    assert ate < 1e-6


if __name__ == "__main__":
    test_compute_ate_is_similarity_invariant()
