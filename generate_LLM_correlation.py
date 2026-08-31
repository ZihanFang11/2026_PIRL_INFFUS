import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch



def source_name(dataset):
    key = dataset.lower()
    if "voc07_3view" in key:
        return "VOC07"
    if "iaprtc12" in key:
        return "IAPRTC12"
    if "corel5k" in key:
        return "corel5k"
    if "mirflickr" in key:
        return "mirflickr"
    return dataset


def load_labels(path):
    with open(path,"r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def calibrate(conf, temperature=2.5, eps=1e-6):
    conf = float(np.clip(float(conf), eps, 1.0 - eps))
    logit = np.log(conf / (1.0 - conf))
    return float(1.0 / (1.0 + np.exp(-logit / temperature)))


def build_matrices(frame):
    required = {"label_1", "label_2", "relation_type", "confidence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Excel is missing columns: {}".format(sorted(missing)))

    labels = sorted(set(frame["label_1"]).union(set(frame["label_2"])))
    label_to_id = {label: index for index, label in enumerate(labels)}
    size = len(labels)
    positive = np.zeros((size, size), dtype=np.float32)
    negative = np.zeros((size, size), dtype=np.float32)
    np.fill_diagonal(positive, 0.0)
    np.fill_diagonal(negative, 0.0)

    for _, row in frame.iterrows():
        i = label_to_id[row["label_1"]]
        j = label_to_id[row["label_2"]]
        relation = str(row["relation_type"]).strip().lower()
        confidence = calibrate(row["confidence"])

        if relation == "likely co-occurring":
            positive[i, j] = confidence
        elif relation == "hierarchical / entailment":
            positive[i, j] = 0.5 * confidence
        else:
            positive[i, j] = 0.0

        negative[i, j] = confidence if relation == "mutually exclusive" else 0.0

    return positive, negative, labels


def generate( dataset, llm, output_dir):
    base = source_name(dataset)
    labels = load_labels( f"./data/label/{base}.txt")
    excel =  f"./labels_relation_llm/{base}_{llm}_relations.xlsx"
    positive, negative, matrix_labels = build_matrices(pd.read_excel(str(excel)))
    label_to_id = {label: index for index, label in enumerate(matrix_labels)}

    missing = [label for label in labels if label not in label_to_id]
    if missing:
        raise ValueError("{}: labels missing from Excel: {}".format(dataset, missing))

    order = [label_to_id[label] for label in labels]
    positive = positive[np.ix_(order, order)]
    negative = negative[np.ix_(order, order)]
    result = {
        "S_pos_llm": torch.from_numpy(positive),
        "S_neg_llm": torch.from_numpy(negative),
        "label_list": labels,
    }


    output_path = f"{output_dir}/{dataset}_{llm}_label_correlation.pt"
    print(f"save to {output_path}")
    torch.save(result, str(output_path))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen-2.5-14B-Instruct")
    parser.add_argument("--output-dir", default= "./LLM_semantic_correlation_matrix/")
    args = parser.parse_args()

    datasets = ["3sources", "emotions", "MIRFlickr_3view", "VOC07_3view",'corel5k_six_view',  'iaprtc12_six_view', ]

    for dataset in datasets:
        generate(
            dataset,
            args.llm,
            args.output_dir
        )


if __name__ == "__main__":
    main()
