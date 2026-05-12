import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 1) CSV 로드
# =========================
beaver_df = pd.read_csv("data/BeaverTails-Evaluation.csv")
hex_df = pd.read_csv("data/hex-phi.csv")

# =========================
# 2) 고정 컬럼 사용
# =========================
hex_prompt_col = "prompt"
hex_category_col = "category"

beaver_prompt_col = "prompt"
beaver_category_col = "category"

# 큰 값이 위로 오게 정렬
hex_counts = hex_df[hex_category_col].value_counts().sort_values(ascending=True)
beaver_counts = beaver_df[beaver_category_col].value_counts().sort_values(ascending=True)


# =========================
# 3) 시각화 함수
# =========================
def plot_dataset_bar(ax, dataset_name, total_prompts, cat_counts):
    categories = cat_counts.index.tolist()
    values = cat_counts.values.tolist()

    bars = ax.barh(categories, values)

    ax.set_title(
        f"{dataset_name}\nTotal Prompts: {total_prompts:,}",
        fontsize=16,
        fontweight="bold",
        pad=18
    )
    ax.set_xlabel("Number of Prompts", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)

    max_val = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(
            v + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(v),
            va="center",
            fontsize=10
        )

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# =========================
# 4) 전체 plot
# =========================
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

plot_dataset_bar(
    axes[0],
    dataset_name="HEx-PHI",
    total_prompts=len(hex_df[hex_prompt_col].dropna()),
    cat_counts=hex_counts
)

plot_dataset_bar(
    axes[1],
    dataset_name="BeaverTails-Evaluation",
    total_prompts=len(beaver_df[beaver_prompt_col].dropna()),
    cat_counts=beaver_counts
)

fig.suptitle("Safety Benchmark Category Distribution", fontsize=22, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("tools/category_plot.png", dpi=300, bbox_inches="tight")
print("Saved to tools/category_plot.png")