# evaluate/make_tables.py
import os
import json
import argparse
import pandas as pd

BEAVER_ID2NAME = {
    0: "animal_abuse",
    1: "child_abuse",
    2: "controversial_topics_politics",
    3: "discrimination_stereotype_injustice",
    4: "drug_abuse_weapons_banned_substance",
    5: "financial_crime_property_crime_theft",
    6: "hate_speech_offensive_language",
    7: "misinformation_ethics_laws_safety",
    8: "non_violent_unethical_behavior",
    9: "privacy_violation",
    10: "self_harm",
    11: "sexually_explicit_adult_content",
    12: "terrorism_organized_crime",
    13: "violence_aiding_abetting_incitement",
}

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_blank(x):
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

# -------------------------
# BeaverTails (flagged/unflagged)
# -------------------------
def summarize_beaver(eval_json_path: str, model_name: str):
    rows = load_json(eval_json_path)  # list[dict]
    df = pd.DataFrame(rows)

    # expected columns: index,prompt,response,category_id,flagged.gpt4
    if "flagged" in df.columns:
        # flagged is dict -> extract
        df["gpt4_flag"] = df["flagged"].apply(lambda d: (d or {}).get("gpt4", None))
    else:
        # sometimes already flattened
        df["gpt4_flag"] = df.get("gpt4", None)

    df["is_blank"] = df["response"].apply(is_blank)
    used = df[~df["is_blank"]].copy()

    used["is_harmful"] = used["gpt4_flag"].astype(str).str.lower().eq("flagged")

    overall = {
        "Model": model_name,
        "Total": int(len(df)),
        "Blank": int(df["is_blank"].sum()),
        "Used": int(len(used)),
        "Harmful_Count": int(used["is_harmful"].sum()),
        "Harm_Rate": float(used["is_harmful"].mean()) if len(used) else float("nan"),
    }

    # category rates
    used["category_name"] = used["category_id"].map(BEAVER_ID2NAME).fillna("unknown")
    cat = (
        used.groupby(["category_id", "category_name"], as_index=False)
        .agg(
            Used=("is_harmful", "size"),
            Harmful_Count=("is_harmful", "sum"),
            Harm_Rate=("is_harmful", "mean"),
        )
        .sort_values("category_id")
    )
    cat["Model"] = model_name
    return overall, cat

# -------------------------
# HEx-PHI (duo_score 1~5)
# -------------------------
def summarize_hexphi(eval_json_path: str, model_name: str):
    rows = load_json(eval_json_path)  # list[dict]
    df = pd.DataFrame(rows)

    # expected columns from your hex_main:
    # category, que, ans, duo_score, duo_reason
    if "ans" not in df.columns and "response" in df.columns:
        df["ans"] = df["response"]

    df["is_blank"] = df["ans"].apply(is_blank)
    used = df[~df["is_blank"]].copy()

    # duo_score should be numeric 1~5
    used["duo_score"] = pd.to_numeric(used["duo_score"], errors="coerce")
    used = used.dropna(subset=["duo_score"]).copy()
    used["is_max_harm"] = used["duo_score"].eq(5)

    overall = {
        "Model": model_name,
        "Total": int(len(df)),
        "Blank": int(df["is_blank"].sum()),
        "Used": int(len(used)),
        "Avg_DuoScore": float(used["duo_score"].mean()) if len(used) else float("nan"),
        "Max5_Rate": float(used["is_max_harm"].mean()) if len(used) else float("nan"),
    }

    # per-category (category is string label)
    if "category" in used.columns:
        cat = (
            used.groupby("category", as_index=False)
            .agg(
                Used=("duo_score", "size"),
                Avg_DuoScore=("duo_score", "mean"),
                Max5_Rate=("is_max_harm", "mean"),
            )
            .sort_values("category")
        )
    else:
        cat = pd.DataFrame(columns=["category", "Used", "Avg_DuoScore", "Max5_Rate"])

    cat["Model"] = model_name
    return overall, cat

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beaver_low", required=True, help="BeaverTails eval json (low)")
    ap.add_argument("--beaver_high", required=True, help="BeaverTails eval json (high)")
    ap.add_argument("--hex_low", required=True, help="HEx-PHI eval json (low)")
    ap.add_argument("--hex_high", required=True, help="HEx-PHI eval json (high)")
    ap.add_argument("--out_dir", default="results/summary_tables", help="output dir")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Beaver
    b_low_overall, b_low_cat = summarize_beaver(args.beaver_low, "Power-Low")
    b_high_overall, b_high_cat = summarize_beaver(args.beaver_high, "Power-High")

    beaver_overall = pd.DataFrame([b_low_overall, b_high_overall])
    beaver_overall["Diff(High-Low)"] = beaver_overall.loc[beaver_overall["Model"]=="Power-High","Harm_Rate"].values[0] - \
                                       beaver_overall.loc[beaver_overall["Model"]=="Power-Low","Harm_Rate"].values[0]
    beaver_overall.to_csv(os.path.join(args.out_dir, "beavertails_overall.csv"), index=False)

    # category merge
    bcat = (
        b_low_cat.merge(
            b_high_cat,
            on=["category_id", "category_name"],
            how="outer",
            suffixes=("_low", "_high"),
        )
        .fillna(0)
        .sort_values("category_id")
    )
    bcat["Diff(High-Low)"] = bcat["Harm_Rate_high"] - bcat["Harm_Rate_low"]
    bcat.to_csv(os.path.join(args.out_dir, "beavertails_by_category.csv"), index=False)

    b_delta_rank = bcat[["category_id","category_name","Harm_Rate_low","Harm_Rate_high","Diff(High-Low)"]].copy()
    b_delta_rank = b_delta_rank.sort_values("Diff(High-Low)", ascending=False)
    b_delta_rank.to_csv(os.path.join(args.out_dir, "beavertails_delta_rank.csv"), index=False)

    # HEx-PHI
    h_low_overall, h_low_cat = summarize_hexphi(args.hex_low, "Power-Low")
    h_high_overall, h_high_cat = summarize_hexphi(args.hex_high, "Power-High")

    hex_overall = pd.DataFrame([h_low_overall, h_high_overall])
    # diff columns
    diff_avg = hex_overall.loc[hex_overall["Model"]=="Power-High","Avg_DuoScore"].values[0] - \
               hex_overall.loc[hex_overall["Model"]=="Power-Low","Avg_DuoScore"].values[0]
    diff_max5 = hex_overall.loc[hex_overall["Model"]=="Power-High","Max5_Rate"].values[0] - \
                hex_overall.loc[hex_overall["Model"]=="Power-Low","Max5_Rate"].values[0]
    hex_overall["Diff_Avg(High-Low)"] = diff_avg
    hex_overall["Diff_Max5(High-Low)"] = diff_max5
    hex_overall.to_csv(os.path.join(args.out_dir, "hexphi_overall.csv"), index=False)

    # category merge
    hcat = (
        h_low_cat.merge(
            h_high_cat,
            on=["category"],
            how="outer",
            suffixes=("_low", "_high"),
        )
        .fillna(0)
        .sort_values("category")
    )
    hcat["Diff_Avg(High-Low)"] = hcat["Avg_DuoScore_high"] - hcat["Avg_DuoScore_low"]
    hcat["Diff_Max5(High-Low)"] = hcat["Max5_Rate_high"] - hcat["Max5_Rate_low"]
    hcat.to_csv(os.path.join(args.out_dir, "hexphi_by_category.csv"), index=False)

    h_delta_rank = hcat[["category","Avg_DuoScore_low","Avg_DuoScore_high","Diff_Avg(High-Low)","Max5_Rate_low","Max5_Rate_high","Diff_Max5(High-Low)"]].copy()
    h_delta_rank = h_delta_rank.sort_values("Diff_Avg(High-Low)", ascending=False)
    h_delta_rank.to_csv(os.path.join(args.out_dir, "hexphi_delta_rank.csv"), index=False)

    print(f"[OK] Saved tables to: {args.out_dir}")
    print(" - beavertails_overall.csv")
    print(" - beavertails_by_category.csv")
    print(" - beavertails_delta_rank.csv")
    print(" - hexphi_overall.csv")
    print(" - hexphi_by_category.csv")
    print(" - hexphi_delta_rank.csv")

if __name__ == "__main__":
    main()
