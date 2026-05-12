import json
import pandas as pd

LOW_PATH = "results/mistral-7b-VIM-beavertails_results_low/gpt-4o_eval_output/beavertails_results.json"
HIGH_PATH = "results/mistral-7b-VIM-beavertails_results_high/gpt-4o_eval_output/beavertails_results.json"


def load_clean(path):
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # blank 제거
    df["is_blank"] = df["response"].fillna("").str.strip() == ""
    blank_count = df["is_blank"].sum()

    df_clean = df[~df["is_blank"]].copy()

    # harmful 여부
    df_clean["harmful"] = df_clean["flagged"].apply(lambda x: x["gpt4"] == "flagged")

    return df, df_clean, blank_count


def summarize(name, df_all, df_clean, blanks):
    print("\n======================")
    print(name)
    print("======================")
    print("Total samples        :", len(df_all))
    print("Blank responses      :", blanks, f"({blanks/len(df_all):.3f})")
    print("Used for evaluation  :", len(df_clean))
    print("Harmful rate         :", round(df_clean["harmful"].mean(), 3))


def category_rate(df_clean):
    return (
        df_clean.groupby("category_id")["harmful"]
        .mean()
        .reset_index()
    )


low_all, low_clean, low_blanks = load_clean(LOW_PATH)
high_all, high_clean, high_blanks = load_clean(HIGH_PATH)

summarize("Power LOW", low_all, low_clean, low_blanks)
summarize("Power HIGH", high_all, high_clean, high_blanks)

# 카테고리별 비교
low_cat = category_rate(low_clean)
high_cat = category_rate(high_clean)

merged = low_cat.merge(high_cat, on="category_id", suffixes=("_low", "_high"))
merged["diff(high-low)"] = merged["harmful_high"] - merged["harmful_low"]

print("\n=== Category Harm Rate (no blanks) ===")
print(merged.to_string(index=False))
