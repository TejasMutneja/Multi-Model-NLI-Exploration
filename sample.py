# Code to inspect label semantics and optionally flip labels

import pandas as pd

#  Loading original train.csv
train = pd.read_csv("train.csv").dropna(subset=["premise", "hypothesis", "label"])
train["label"] = train["label"].astype(int)

# Two random examples for each numeric label
print("=== Sample Examples by Original Numeric Label ===\n")
for lbl in sorted(train["label"].unique()):
    print(f"--- Label {lbl} ---")
    samples = train[train["label"] == lbl].sample(2, random_state=lbl)
    for _, row in samples.iterrows():
        print(f"Premise   : {row['premise']}")
        print(f"Hypothesis: {row['hypothesis']}\n")
    print()

#  Flipping labels as mapping is reversed
#     0->2, 1->1, 2->0
flip_map = {0: 2, 1: 1, 2: 0}
train["label_flipped"] = train["label"].map(flip_map)

# Comparing distributions before and after flip
orig_dist = train["label"].value_counts().sort_index()
flip_dist = train["label_flipped"].value_counts().sort_index()
print("Original label distribution:\n", orig_dist, "\n")
print("Flipped label distribution:\n", flip_dist)
