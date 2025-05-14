
import os
import csv
import re
import random
import openai
from tqdm import tqdm
from sklearn.metrics import accuracy_score





openai.api_key = os.getenv("OPENAI_API_KEY")

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
INPUT   = "val.csv"               # validation file must include a 'label' column
MODEL   = "gpt-4o-mini"
N       = 100                     # number of random examples to evaluate
SEED    = 42                     # for reproducibility

# ────────────────────────────────────────────────────────────────
# Prompt template
# ────────────────────────────────────────────────────────────────
PROMPT = """
Task
You will be given two short texts: a Premise and a Hypothesis. Decide whether the Hypothesis is
contradicted by the Premise (label 0), logically unrelated / possible but not guaranteed (label 1 = neutral), 
or logically entailed by the Premise (label 2).
Return only the single digit 0, 1, or 2—no extra words.

Decision Guide

0 (Contradiction)    The Hypothesis cannot be true if the Premise is true.
1 (Neutral)          Both texts can be true at the same time, but the Premise does not guarantee the Hypothesis.
2 (Entailment)       If the Premise is true, the Hypothesis must also be true.

Format
Premise: {premise_text}
Hypothesis: {hypothesis_text}
Answer:
"""

# ────────────────────────────────────────────────────────────────
# Loading all rows, then sample N randomly
# ────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    all_rows = list(reader)

random.seed(SEED)
rows = random.sample(all_rows, N)

# ────────────────────────────────────────────────────────────────
#  Inference loop
# ────────────────────────────────────────────────────────────────
y_true = []
y_pred = []

for row in tqdm(rows, desc=f"GPT-4 random {N}", unit="row"):
    prompt = PROMPT.format(
        premise_text=row["premise"].strip(),
        hypothesis_text=row["hypothesis"].strip()
    ).strip()

    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4
    )

    text = resp.choices[0].message.content.strip().lower()
    m = re.search(r"\b([012])\b", text)
    pred = int(m.group(1)) if m else 1  # fallback to neutral

    y_pred.append(pred)
    y_true.append(int(row["label"]))

# ────────────────────────────────────────────────────────────────
#  Accuracy
# ────────────────────────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy on {N} random examples: {acc:.3f}")
