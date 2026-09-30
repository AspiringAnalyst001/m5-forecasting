import pandas as pd

local = pd.read_csv("results/backtest_CA_1.csv")            # fresh reruns
nhits = pd.read_csv("results/backtest_allCA_nhits_only.csv")  # from Colab, still intact

merged = pd.concat([local, nhits], ignore_index=True)
merged = merged.drop_duplicates(subset=["model", "fold"], keep="last")
merged.to_csv("results/backtest_CA_1.csv", index=False)
print(merged[["model", "fold", "wrmsse"]].sort_values(["model", "fold"]).to_string(index=False))