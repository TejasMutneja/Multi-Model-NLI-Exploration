import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load & clean
df = pd.read_csv("train.csv").dropna(subset=["premise","hypothesis","label"])
df["label"] = df["label"].astype(int)

# 2. Flip 0↔2 so that 0=Contra,1=Neutral,2=Entail
flip_map = {0:2, 1:1, 2:0}
df["label"] = df["label"].map(flip_map)

# 3. Stratified split 90/10
train_df, val_df = train_test_split(
    df, test_size=0.10, stratify=df["label"], random_state=42
)

# 4. Save out val.csv (and optionally train.csv)
val_df.to_csv("val.csv", index=False)
print(f"Saved val.csv with {len(val_df)} rows (labels flipped)")
