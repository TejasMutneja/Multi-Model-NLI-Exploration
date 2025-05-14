

import csv
import requests
import subprocess
import json
import re
from tqdm import tqdm
from collections import Counter

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
INPUT    = "val.csv"                # or "test.csv" for final run
OUTPUT   = "val_preds.csv"
GEMMA    = "gemma3:12b"             # model for self-consistency
LLAMA    = "llama3:8b"              # model for single-shot vote
K        = 5                        # number of self-consistency samples
TEMP     = 0.7                      # temperature for sampling

# ────────────────────────────────────────────────────────────────
# Few-shot examples (multilingual & multi-label)
# ────────────────────────────────────────────────────────────────
few_shot = [
    ("She did not reply.", "She was silent.", "2"),
    ("He forgot to study.", "He performed well.", "0"),
    ("They hear the music.", "They are listening to it.", "1"),
    ("Il pleuvait toute la journée.", "Le sol était mouillé.", "2"),
    ("لا يوجد ماء في الصحراء.", "الصحراء مليئة بالماء.", "0"),
    ("เด็กกำลังวาดรูป.", "เด็กกำลังวิ่งเล่น.", "1"),
]

# ────────────────────────────────────────────────────────────────
# Majority-vote helper
# ────────────────────────────────────────────────────────────────
def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]

# ────────────────────────────────────────────────────────────────
# Prompt builder (same original prompt + few-shot)
# ────────────────────────────────────────────────────────────────
def build_prompt(premise, hypothesis):
    lines = [
        "Task",
        "You will be given two short texts: a Premise and a Hypothesis. Decide whether the Hypothesis is "
        "contradicted by the Premise (label 0), logically unrelated / possible but not guaranteed (label 1 = neutral), "
        "or logically entailed by the Premise (label 2). Return only the single digit 0, 1, or 2—no extra words.",
        "",
        "Decision Guide",
        "0 (Contradiction)    The Hypothesis cannot be true if the Premise is true.",
        "1 (Neutral)          Both texts can be true at the same time, but the Premise does not guarantee the Hypothesis.",
        "2 (Entailment)       If the Premise is true, the Hypothesis must also be true.",
        ""
    ]
    # inject few-shot examples
    for idx, (p, h, lbl) in enumerate(few_shot, start=1):
        lines += [
            f"Example {idx}:",
            f"Premise:   {p}",
            f"Hypothesis: {h}",
            f"Answer:    {lbl}",
            ""
        ]
    # actual instance
    lines += [
        f"Premise:   {premise}",
        f"Hypothesis: {hypothesis}",
        "Answer:"
    ]
    return "\n".join(lines)

# ────────────────────────────────────────────────────────────────
# Batch inference with ensemble
# ────────────────────────────────────────────────────────────────
with open(INPUT, newline="", encoding="utf-8") as fin, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["id", "prediction"])

    for row in tqdm(reader, desc="Ensemble", unit="row"):
        prompt = build_prompt(row['premise'], row['hypothesis'])

        # 1) Self-consistency on GEMMA
        gemma_preds = []
        for _ in range(K):
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": GEMMA, "prompt": prompt, "options": {"temperature": TEMP, "num_predict": 1}}
            )
            resp_text = None
            for line in r.text.splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in obj:
                    resp_text = obj["response"]
                    break
            out = (resp_text or "").strip().lower()
            m = re.search(r"\b([012])\b", out)
            gemma_preds.append(m.group(1) if m else "1")

        # 2) Single-shot vote from LLAMA
        proc = subprocess.run(
            ["ollama", "run", LLAMA, prompt], capture_output=True, text=True
        )
        text = proc.stdout.strip().lower()
        m2 = re.search(r"\b([012])\b", text)
        llama_pred = m2.group(1) if m2 else "1"

        # 3) Majority vote across both
        all_preds = gemma_preds + [llama_pred]
        final = majority_vote(all_preds)

        writer.writerow([row["id"], final])

print(f"✅ Wrote predictions to {OUTPUT}")
