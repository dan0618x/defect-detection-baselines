"""Compare a rule-based OpenCV detector, a random forest and a CNN under standard and harsh lighting."""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from defect_detection.config import DEFECT_TYPES, N_TEST, N_TRAIN, SEED
from defect_detection.dataset import generate_dataset
from defect_detection.metrics import binary_scores
from defect_detection.models import predict_cnn, train_cnn, train_random_forest
from defect_detection.rule_based import features_of, predict_rule, tune_rule

MS_PER_SECOND = 1000
SAMPLES_PER_ROW = 6
CHART_Y_MAX = 1.15           # headroom so the legend does not cover bars
SAMPLES_PATH = os.path.join("assets", "samples.png")
F1_PATH = os.path.join("assets", "f1_comparison.png")
SEED_OFFSETS = {"train_standard": 1, "train_harsh": 2, "test_standard": 3, "test_harsh": 4}


def build_datasets():
    half = N_TRAIN // 2
    data = {
        "train_standard": generate_dataset(N_TRAIN, "standard", SEED + SEED_OFFSETS["train_standard"]),
        "train_harsh": generate_dataset(half, "harsh", SEED + SEED_OFFSETS["train_harsh"]),
        "test_standard": generate_dataset(N_TEST, "standard", SEED + SEED_OFFSETS["test_standard"]),
        "test_harsh": generate_dataset(N_TEST, "harsh", SEED + SEED_OFFSETS["test_harsh"]),
    }
    std_images, std_labels, std_kinds = data["train_standard"]
    harsh_images, harsh_labels, harsh_kinds = data["train_harsh"]
    data["train_mixed"] = (np.concatenate([std_images[:half], harsh_images]),
                           np.concatenate([std_labels[:half], harsh_labels]),
                           std_kinds[:half] + harsh_kinds)
    return data


def fit_methods(images, labels):
    """Return {name: predict(images) -> labels} for the three methods."""
    features = features_of(images)
    solidity_min, dark_ratio_max = tune_rule(features, labels)
    forest = train_random_forest(features, labels)
    cnn, _ = train_cnn(images, labels)
    return {
        f"rule (solidity<{solidity_min:.3f} or dark>{dark_ratio_max:.4f})":
            lambda x: predict_rule(features_of(x), solidity_min, dark_ratio_max),
        "random forest on OpenCV features": lambda x: forest.predict(features_of(x)),
        "CNN on pixels (Keras)": lambda x: predict_cnn(cnn, x),
    }


def evaluate(methods, data, train_name):
    rows = []
    for name, predict in methods.items():
        row = {"method": name, "train": train_name}
        for test_name in ["test_standard", "test_harsh"]:
            images, labels, _ = data[test_name]
            start_time = time.perf_counter()
            predictions = predict(images)
            row[test_name] = binary_scores(labels, predictions)
            row[test_name]["ms_per_image"] = (time.perf_counter() - start_time) * MS_PER_SECOND / len(images)
            row[test_name]["recall_by_type"] = recall_by_type(data[test_name][2], predictions)
        rows.append(row)
    return rows


def recall_by_type(kinds, predictions):
    """Share of each defect type that was flagged (for normal parts: share wrongly flagged)."""
    result = {}
    for kind in DEFECT_TYPES + [None]:
        hits = [prediction for name, prediction in zip(kinds, predictions) if name == kind]
        result[kind or "normal (false alarm)"] = float(np.mean(hits))
    return result


def print_type_table(rows):
    names = list(rows[0]["test_harsh"]["recall_by_type"])
    print("\n| method | trained on | " + " | ".join(names) + " |")
    print("|---|---|" + "---:|" * len(names))
    for row in rows:
        cells = [f"{row['test_harsh']['recall_by_type'][name]:.2f}" for name in names]
        print(f"| {row['method'].split(' (')[0]} | {row['train']} | " + " | ".join(cells) + " |")


def print_table(rows):
    print("| method | trained on | standard P / R / F1 | harsh P / R / F1 | ms / image |")
    print("|---|---|---|---|---:|")
    for row in rows:
        cells = [f"{row[t]['precision']:.2f} / {row[t]['recall']:.2f} / **{row[t]['f1']:.2f}**"
                 for t in ["test_standard", "test_harsh"]]
        print(f"| {row['method']} | {row['train']} | {cells[0]} | {cells[1]} | {row['test_standard']['ms_per_image']:.2f} |")


def save_samples(data):
    figure, axes = plt.subplots(2, SAMPLES_PER_ROW, figsize=(12, 4.5))
    for row_index, test_name in enumerate(["test_standard", "test_harsh"]):
        images, _, kinds = data[test_name]
        for column in range(SAMPLES_PER_ROW):
            axes[row_index, column].imshow(cv2.cvtColor(images[column], cv2.COLOR_BGR2RGB))
            axes[row_index, column].set_title(f"{test_name[5:]}: {kinds[column] or 'normal'}", fontsize=9)
            axes[row_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(SAMPLES_PATH)
    plt.close(figure)


def save_f1_chart(rows):
    labels = [f"{row['method'].split(' (')[0]}\n[{row['train']}]" for row in rows]
    positions = np.arange(len(rows))
    width = 0.4
    plt.figure(figsize=(11, 4.5))
    plt.bar(positions - width / 2, [row["test_standard"]["f1"] for row in rows], width, label="standard lighting")
    plt.bar(positions + width / 2, [row["test_harsh"]["f1"] for row in rows], width, label="harsh lighting")
    plt.xticks(positions, labels, fontsize=8)
    plt.ylabel("F1 (defective class)")
    plt.ylim(0, CHART_Y_MAX)
    plt.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig(F1_PATH)
    plt.close()


def main():
    data = build_datasets()
    save_samples(data)
    rows = []
    for train_name in ["train_standard", "train_mixed"]:
        images, labels, _ = data[train_name]
        start_time = time.perf_counter()
        methods = fit_methods(images, labels)
        print(f"fitted on {train_name} ({len(labels)} images) in {time.perf_counter() - start_time:.1f} s")
        rows.extend(evaluate(methods, data, train_name[6:]))
    print_table(rows)
    print_type_table(rows)
    save_f1_chart(rows)
    print(f"saved {SAMPLES_PATH} and {F1_PATH}")


if __name__ == "__main__":
    main()
