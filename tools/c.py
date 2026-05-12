import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1) load
# =========================
cat_path = "/home/jaesang02/Unintended-Harms-LLM-reproduction/results/summary_tables/beavertails_by_category.csv"
df_cat = pd.read_csv(cat_path)

# Δ가 큰 순서대로 정렬
df_cat = df_cat.sort_values("Diff(High-Low)", ascending=False).reset_index(drop=True)

# x축 위치
x = np.arange(len(df_cat))
width = 0.38

# =========================
# 2) 카테고리별 Harm_Rate: low vs high
# =========================
plt.figure(figsize=(14, 6))
plt.bar(x - width/2, df_cat["Harm_Rate_low"], width, label="Power-Low")
plt.bar(x + width/2, df_cat["Harm_Rate_high"], width, label="Power-High")

plt.xticks(x, df_cat["category_name"], rotation=45, ha="right")
plt.ylabel("Harm Rate (flagged / used)")
plt.title("BeaverTails Category-wise Harm Rate: Power-Low vs Power-High")
plt.legend()
plt.tight_layout()
plt.savefig("beaver_category_low_vs_high.png", dpi=300)
plt.close()

# =========================
# 3) 카테고리별 Δ Harm Rate
# =========================
plt.figure(figsize=(12, 6))
bars = plt.barh(df_cat["category_name"], df_cat["Diff(High-Low)"])
plt.xlabel("Δ Harm Rate (High - Low)")
plt.title("BeaverTails Category-wise Difference in Harm Rate")
plt.gca().invert_yaxis()

for bar, v in zip(bars, df_cat["Diff(High-Low)"]):
    offset = max(df_cat["Diff(High-Low)"].abs().max() * 0.01, 0.002)
    x_text = v + offset if v >= 0 else v - offset
    ha = "left" if v >= 0 else "right"
    plt.text(
        x_text,
        bar.get_y() + bar.get_height()/2,
        f"{v:.3f}",
        va="center",
        ha=ha,
        fontsize=10
    )

plt.tight_layout()
plt.savefig("beaver_category_delta.png", dpi=300)
plt.close()

# =========================
# 4) Top-3 / Bottom-3 따로
# =========================
df_top = df_cat.sort_values("Diff(High-Low)", ascending=False).head(3)
df_bottom = df_cat.sort_values("Diff(High-Low)", ascending=True).head(3)

plt.figure(figsize=(8, 4.5))
bars = plt.barh(df_top["category_name"], df_top["Diff(High-Low)"])
plt.xlabel("Δ Harm Rate (High - Low)")
plt.title("BeaverTails Top-3 Δ Categories")
plt.gca().invert_yaxis()

for bar, v in zip(bars, df_top["Diff(High-Low)"]):
    plt.text(v + 0.002, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center")

plt.tight_layout()
plt.savefig("beaver_top3_delta.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4.5))
bars = plt.barh(df_bottom["category_name"], df_bottom["Diff(High-Low)"])
plt.xlabel("Δ Harm Rate (High - Low)")
plt.title("BeaverTails Bottom-3 Δ Categories")
plt.gca().invert_yaxis()

for bar, v in zip(bars, df_bottom["Diff(High-Low)"]):
    plt.text(v - 0.002, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", ha="right")

plt.tight_layout()
plt.savefig("beaver_bottom3_delta.png", dpi=300)
plt.close()

# =========================
# 5) 한 장짜리 combined plot
# =========================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# low vs high
axes[0].bar(x - width/2, df_cat["Harm_Rate_low"], width, label="Power-Low")
axes[0].bar(x + width/2, df_cat["Harm_Rate_high"], width, label="Power-High")
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_cat["category_name"], rotation=45, ha="right")
axes[0].set_ylabel("Harm Rate")
axes[0].set_title("BeaverTails Category-wise Harm Rate: Power-Low vs Power-High")
axes[0].legend()

# delta
bars = axes[1].barh(df_cat["category_name"], df_cat["Diff(High-Low)"])
axes[1].set_xlabel("Δ Harm Rate (High - Low)")
axes[1].set_title("BeaverTails Category-wise Difference in Harm Rate")
axes[1].invert_yaxis()

for bar, v in zip(bars, df_cat["Diff(High-Low)"]):
    offset = max(df_cat["Diff(High-Low)"].abs().max() * 0.01, 0.002)
    x_text = v + offset if v >= 0 else v - offset
    ha = "left" if v >= 0 else "right"
    axes[1].text(
        x_text,
        bar.get_y() + bar.get_height()/2,
        f"{v:.3f}",
        va="center",
        ha=ha,
        fontsize=10
    )

plt.tight_layout()
plt.savefig("beaver_category_combined.png", dpi=300)
plt.close()

print("Saved:")
print("- beaver_category_low_vs_high.png")
print("- beaver_category_delta.png")
print("- beaver_top3_delta.png")
print("- beaver_bottom3_delta.png")
print("- beaver_category_combined.png")