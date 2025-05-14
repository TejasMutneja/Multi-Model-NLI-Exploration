#!/usr/bin/env python3
import csv, requests, json, re
from tqdm import tqdm
from collections import Counter

INPUT  = "val.csv"
OUTPUT = "val_preds.csv"
MODEL  = "gemma3:12b"
K      = 5
TEMP   = 0.7

few_shot = [
    ("She did not reply.","She was silent.","2"),
    ("He forgot to study.","He performed well.","0"),
    ("They hear the music.","They are listening to it.","1"),
    ("Il pleuvait toute la journée.","Le sol était mouillé.","2"),
    ("لا يوجد ماء في الصحراء.","الصحراء مليئة بالماء.","0"),
    ("เด็กกำลังวาดรูป.","เด็กกำลังวิ่งเล่น.","1"),
]

def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]

with open(INPUT, newline="", encoding="utf-8") as fin, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["id","prediction"])

    for row in tqdm(reader, desc="Few‐shot + SC", unit="row"):
        #  Build the prompt lines properly, *without* embedding the actual instance yet
        prompt_lines = [
            "Task",
            "You will be given two short texts: a Premise and a Hypothesis. Decide whether the Hypothesis is "
            "contradicted by the Premise (label 0), logically unrelated (label 1 = neutral), "
            "or logically entailed by the Premise (label 2). Return only the single digit 0, 1, or 2—no extra words.",
            "",
            "Decision Guide",
            "0 (Contradiction)    The Hypothesis cannot be true if the Premise is true.",
            "1 (Neutral)          Both texts can be true at the same time, but the Premise does not guarantee the Hypothesis.",
            "2 (Entailment)       If the Premise is true, the Hypothesis must also be true.",
            ""
        ]

        #  Inject few-shot examples *before* the actual instance
        for idx, (prem, hyp, lbl) in enumerate(few_shot, start=1):
            prompt_lines += [
                f"Example {idx}:",
                f"Premise:   {prem}",
                f"Hypothesis:{hyp}",
                f"Answer:    {lbl}",
                ""
            ]

        #  Now append the actual example
        prompt_lines += [
            f"Premise:   {row['premise']}",
            f"Hypothesis:{row['hypothesis']}",
            "Answer:"
        ]

        prompt = "\n".join(prompt_lines)

        # Sample K times and collect predictions
        samples = []
        for _ in range(K):
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "options": {"temperature": TEMP, "num_predict": 1}
                }
            )
            # parse streaming JSON lines
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
            samples.append(m.group(1) if m else "1")

        #  Final vote
        writer.writerow([row["id"], majority_vote(samples)])

print(f"✅ Wrote few‐shot self‐consistency predictions to {OUTPUT}")
