import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

gt   = pd.read_csv("val.csv")
pred = pd.read_csv("val_preds.csv")
df   = gt.merge(pred, on="id")

y_true = df.label.astype(int)
y_pred = df.prediction.astype(int)

print("Acc   :", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print(classification_report(y_true, y_pred,
      target_names=["contra","neutral","enta"]))
