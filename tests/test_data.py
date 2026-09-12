"""Tests for RAF-DB dataset splitting logic (no image files required)"""

from sit_fer.data.raf_dataset import data_split, target_read


def _write_label_file(tmp_path, labels):
    path = tmp_path / "labels.txt"
    lines = [f"img_{i}.jpg {label}" for i, label in enumerate(labels)]
    path.write_text("\n".join(lines))
    return str(path)


def test_target_read_parses_labels(tmp_path):
    labels = [0, 1, 2, 3, 4, 5, 6]
    path = _write_label_file(tmp_path, labels)

    assert target_read(path) == labels


def test_data_split_partitions_all_indices(tmp_path):
    # 20 samples per class, 7 classes = 140 total
    labels = [c for c in range(7) for _ in range(20)]
    path = _write_label_file(tmp_path, labels)

    n_labeled = 70
    labeled_idxs, unlabeled_idxs = data_split(path, n_labeled, num_classes=7)

    assert len(set(labeled_idxs) & set(unlabeled_idxs)) == 0
    assert len(labeled_idxs) + len(unlabeled_idxs) == len(labels)
