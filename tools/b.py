import pandas as pd
import matplotlib.pyplot as plt

# 1) overall metrics
overall_path = "/home/jaesang02/Unintended-Harms-LLM-reproduction/results/summary_tables/hexphi_overall.csv"
df_overall = pd.read_csv(overall_path)

# Avg DuoScore
plt.figure()
plt.bar(df_overall["Model"], df_overall["Avg_DuoScore"])
plt.ylabel("Avg duo_score (1~5)")
plt.title("HEx-PHI Avg Harmfulness Score: Power-Low vs Power-High")
plt.tight_layout()
plt.savefig("hex_overall_avg.png", dpi=300)

# Max5 rate
plt.figure()
plt.bar(df_overall["Model"], df_overall["Max5_Rate"])
plt.ylabel("Rate of duo_score == 5")
plt.title("HEx-PHI Max-5 (Severe) Rate: Power-Low vs Power-High")
plt.tight_layout()
plt.savefig("hex_overall_max5.png", dpi=300)

# 2) Top-3 delta (Avg)
cat_path = "/home/jaesang02/Unintended-Harms-LLM-reproduction/results/summary_tables/hexphi_by_category.csv"
df_cat = pd.read_csv(cat_path)

df_top = df_cat.sort_values("Diff_Avg(High-Low)", ascending=False).head(3)

plt.figure()
plt.barh(df_top["category"], df_top["Diff_Avg(High-Low)"])
plt.xlabel("Δ Avg duo_score (High - Low)")
plt.title("HEx-PHI Top-3 Δ Categories (Avg)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("hex_top3_delta_avg.png", dpi=300)


