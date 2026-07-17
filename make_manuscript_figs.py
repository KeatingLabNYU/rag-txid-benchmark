"""
make_manuscript_figs.py
=======================
Generates all 7 publication-quality figures for PLOS Digital Health submission.
Run from the LLM_paper_Claude_Opus_max directory, or set BASE below.

Output: figures/manuscript/Fig{1-7}_*.{png,pdf}  @ 600 dpi
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import wilcoxon, t as tdist

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "figures", "manuscript")
os.makedirs(OUT, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE  = "#2C6E9B"
RED   = "#C0392B"
GREEN = "#1A8A5A"
GRAY  = "#999999"
LGRAY = "#DDDDDD"
FS    = 11   # unified axis label / tick font size

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     FS,
    "axes.titlesize":     12,
    "xtick.labelsize":    FS,
    "ytick.labelsize":    FS,
    "legend.fontsize":    10,
})

def ph(ax, label, title):
    """Panel header: bold letter + bold title in axes-transform coords."""
    ax.text(-0.10, 1.10, label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom")
    ax.text( 0.00, 1.03, title, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

def boxleg(ax, handles, **kw):
    """Legend with white box and light border."""
    leg = ax.legend(handles=handles, frameon=True, facecolor="white",
                    edgecolor="#AAAAAA", framealpha=1.0, borderpad=0.7, **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    dpi=600, bbox_inches="tight", facecolor="white")
    print(f"  saved {name}")
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────────────────────
df  = pd.read_csv(os.path.join(BASE,
        "all_models_rag_non_rag_judged.csv"))
exp = pd.read_csv(os.path.join(BASE, "expert_evaluations_linked_long.csv"))
t4  = pd.read_csv(os.path.join(BASE, "tables", "Table4_error_taxonomy.csv"))

DIMS = ["correctness","completeness","reasoning","relevance","clarity","consistency"]
df["composite"] = df[DIMS].mean(axis=1)
rag_df   = df[df["RAG_MODEL"] == True]
norag_df = df[df["RAG_MODEL"] == False]

DIM_QVALS = {
    "completeness": 1e-10, "reasoning": 1e-8, "clarity": 1e-7,
    "relevance": 1e-7, "correctness": 0.003, "consistency": 0.04,
}
def stars(q): return "***" if q<0.001 else "**" if q<0.01 else "*" if q<0.05 else "ns"


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Primary degradation + human validation
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1 …")

dim_means = {d: (norag_df[d].mean(), rag_df[d].mean()) for d in DIMS}
order = sorted(DIMS, key=lambda d: dim_means[d][0] - dim_means[d][1])
dlabels = {"correctness":"Correctness","completeness":"Completeness",
           "reasoning":"Reasoning","relevance":"Relevance",
           "clarity":"Clarity","consistency":"Consistency"}

exp_rag   = exp[exp["mode"] == "RAG"]
exp_norag = exp[exp["mode"] == "No RAG"]
exp_metrics = {
    "Accuracy":   (exp_norag["accuracy"].mean(),   exp_rag["accuracy"].mean()),
    "Usefulness": (exp_norag["usefulness"].mean(),  exp_rag["usefulness"].mean()),
}
acc_p = wilcoxon(exp_norag.sort_values("case_code")["accuracy"].values,
                 exp_rag.sort_values("case_code")["accuracy"].values).pvalue
use_p = wilcoxon(exp_norag.sort_values("case_code")["usefulness"].values,
                 exp_rag.sort_values("case_code")["usefulness"].values).pvalue

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
fig.subplots_adjust(wspace=0.40, top=0.88, bottom=0.10)

# Panel A
ax = axes[0]
for i, d in enumerate(order):
    n_m, r_m = dim_means[d]
    ax.plot([n_m, r_m], [i, i], color=LGRAY, lw=1.2, zorder=1)
    ax.scatter(n_m, i, color=BLUE, s=60, zorder=3)
    ax.scatter(r_m, i, color=RED,  s=60, zorder=3)
    s = stars(DIM_QVALS[d])
    if s != "ns":
        ax.text(max(n_m, r_m)+0.04, i, s, va="center", ha="left", fontsize=9, color="#444")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([dlabels[d] for d in order], fontsize=FS)
ax.set_xlabel("Mean score (1–5 scale)", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.set_xlim(2.4, 6.0)   # extended right so legend clears all data
ax.xaxis.grid(True, color=LGRAY, lw=0.6, zorder=0); ax.set_axisbelow(True)
boxleg(ax, [mpatches.Patch(color=BLUE, label="No-RAG"),
            mpatches.Patch(color=RED,  label="RAG")], loc="upper right")
ph(ax, "A", "LLM-judge ratings by condition")

# Panel B
ax = axes[1]
for i, m in enumerate(["Accuracy", "Usefulness"]):
    n_m, r_m = exp_metrics[m]
    p_val = acc_p if m == "Accuracy" else use_p
    ax.plot([n_m, r_m], [i, i], color=LGRAY, lw=1.2, zorder=1)
    ax.scatter(n_m, i, color=BLUE, s=65, zorder=3)
    ax.scatter(r_m, i, color=RED,  s=65, zorder=3)
    s = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.text(max(n_m, r_m)+0.07, i, s, va="center", ha="left", fontsize=9, color="#444")
    ax.text(n_m, i-0.15, f"{n_m:.2f}", ha="center", va="top", fontsize=9, color=BLUE)
    ax.text(r_m, i-0.15, f"{r_m:.2f}", ha="center", va="top", fontsize=9, color=RED)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Accuracy", "Usefulness"], fontsize=FS)
ax.set_xlabel("Mean rating (1–5 scale)", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.set_xlim(2.5, 5.0); ax.set_ylim(-0.5, 1.5)
ax.xaxis.grid(True, color=LGRAY, lw=0.6, zorder=0); ax.set_axisbelow(True)
ph(ax, "B", "Blinded expert ratings by condition")
save(fig, "Fig1_primary_degradation")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Per-model RAG effect
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2 …")

model_stats = df.groupby(["model_name", "RAG_MODEL"]).agg(
    composite=("composite","mean")).reset_index()
piv = model_stats.pivot_table(
    index="model_name", columns="RAG_MODEL", values="composite").reset_index()
piv.columns = ["model_name","norag","rag"]
piv["delta"] = piv["rag"] - piv["norag"]
piv = piv.dropna().sort_values("norag")
n_models = len(piv)

fig, ax = plt.subplots(figsize=(7.5, max(8, n_models*0.24)))
fig.subplots_adjust(top=0.94, bottom=0.06)
colors = [GREEN if d > 0 else RED for d in piv["delta"]]
ys = np.arange(n_models)
ax.barh(ys, piv["delta"].values, color=colors, height=0.65, edgecolor="none")
ax.axvline(0, color="#333", lw=1.0, zorder=5)
ax.set_yticks(ys)
ax.set_yticklabels(piv["model_name"].values, fontsize=FS)
ax.set_xlabel("RAG − No-RAG composite quality", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.xaxis.grid(True, color=LGRAY, lw=0.5, zorder=0); ax.set_axisbelow(True)
for idx, row in piv.iterrows():
    if row["delta"] > 0:
        i = list(piv.index).index(idx)
        ax.text(row["delta"]+0.01, i, row["model_name"],
                va="center", ha="left", fontsize=8.5, color=GREEN)
ax.spines["left"].set_visible(False); ax.tick_params(left=False)
ax.set_title("Per-model effect of the evaluated RAG pipeline on composite quality",
             fontweight="bold", pad=8)
save(fig, "Fig2_per_model_effect")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Human preference + convergent validity
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3 …")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    paired = exp.groupby(["case_code","evaluator_name"], group_keys=False).apply(
        lambda g: pd.Series({
            "preferred_side": g["preference"].iloc[0],
            "RAG_side":   g.loc[g["mode"]=="RAG",    "side"].iloc[0] if (g["mode"]=="RAG").any()    else None,
            "NoRAG_side": g.loc[g["mode"]=="No RAG", "side"].iloc[0] if (g["mode"]=="No RAG").any() else None,
        })
    ).reset_index()

def map_pref(row):
    if row["preferred_side"] == "Tie":           return "Tie"
    if row["preferred_side"] == row["NoRAG_side"]: return "No-RAG"
    if row["preferred_side"] == row["RAG_side"]:   return "RAG"
    return "Tie"

paired["preferred_condition"] = paired.apply(map_pref, axis=1)
pc = paired["preferred_condition"].value_counts()
n_norag = int(pc.get("No-RAG", 0))
n_rag   = int(pc.get("RAG",    0))
n_tie   = int(pc.get("Tie",    0))
total   = n_norag + n_rag + n_tie

# Convergent validity — aggregate to (case, model, condition) level
exp["RAG_MODEL"] = exp["mode"].map({"RAG": True, "No RAG": False})
llm_corr = df.groupby(["item_id","model_name","RAG_MODEL"])["correctness"].mean().reset_index()
llm_corr.columns = ["case_code","model_name","RAG_MODEL","llm_correctness"]
merged_cv = exp.merge(llm_corr, on=["case_code","model_name","RAG_MODEL"], how="inner")
agg_cv = merged_cv.groupby(["case_code","model_name","RAG_MODEL"]).agg(
    expert_acc=("accuracy","mean"),
    llm_corr=("llm_correctness","first")).reset_index()
rho, pval = stats.spearmanr(agg_cv["expert_acc"], agg_cv["llm_corr"])
m_cv, b_cv = np.polyfit(agg_cv["llm_corr"], agg_cv["expert_acc"], 1)
pt_c  = [BLUE if not r else RED for r in agg_cv["RAG_MODEL"]]
n_cv  = len(agg_cv)
resid = agg_cv["expert_acc"].values - (m_cv*agg_cv["llm_corr"].values + b_cv)
ci_cv = tdist.ppf(0.975, n_cv-2) * np.std(resid) / np.sqrt(n_cv)
x_line = np.linspace(agg_cv["llm_corr"].min(), agg_cv["llm_corr"].max(), 100)

fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
fig.subplots_adjust(wspace=0.44, top=0.88, bottom=0.10)

# Panel A
ax = axes[0]
left = 0
for val, col in zip([n_rag, n_tie, n_norag], [RED, GRAY, BLUE]):
    ax.barh(0, val, left=left, color=col, height=0.4)
    if val > 0:
        ax.text(left+val/2, 0, str(val), va="center", ha="center",
                fontsize=11, fontweight="bold", color="white")
    left += val
ax.set_xlim(0, total); ax.set_yticks([])
ax.set_xlabel("Number of paired comparisons", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
boxleg(ax, [mpatches.Patch(color=BLUE, label="No-RAG preferred"),
            mpatches.Patch(color=GRAY, label="Tie"),
            mpatches.Patch(color=RED,  label="RAG preferred")], loc="lower right")
ax.spines["left"].set_visible(False)
ph(ax, "A", "Expert blinded head-to-head preference")

# Panel B
ax = axes[1]
ax.scatter(agg_cv["llm_corr"], agg_cv["expert_acc"],
           c=pt_c, s=50, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.3)
ax.plot(x_line, m_cv*x_line+b_cv, color="#555", lw=1.4, zorder=4)
ax.fill_between(x_line, m_cv*x_line+b_cv-ci_cv, m_cv*x_line+b_cv+ci_cv,
                color="#555", alpha=0.12, zorder=2)
ax.text(0.05, 0.90, f"Spearman ρ = {rho:.2f}\nP = {pval:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCC", alpha=0.9))
ax.set_xlabel("LLM-judged correctness (mean)", fontsize=FS)
ax.set_ylabel("Expert accuracy rating (mean)", fontsize=FS)
ax.tick_params(axis="both", labelsize=FS)
boxleg(ax, [mpatches.Patch(color=BLUE, label="No-RAG"),
            mpatches.Patch(color=RED,  label="RAG")], loc="lower right")
ph(ax, "B", "Convergent validity of LLM-judge scores")
save(fig, "Fig3_human_validation")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Competence paradox
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4 …")

piv2 = model_stats.pivot_table(
    index="model_name", columns="RAG_MODEL", values="composite").reset_index()
piv2.columns = ["model_name","norag","rag"]
piv2["delta"] = piv2["rag"] - piv2["norag"]
piv2 = piv2.dropna().reset_index(drop=True)
r_corr, r_p     = stats.pearsonr(piv2["norag"], piv2["delta"])
rho_c,  rho_p   = stats.spearmanr(piv2["norag"], piv2["delta"])

LABEL_MODELS = {"DeepSeek-R1 8B","Nemotron Nano 12B","Jamba 1.5 Mini",
                "Kimi K2","Jamba 1.5 Large","Mistral Large 3","GPT-OSS 120B"}

# Per-model label offsets: (dx, dy, ha)
LABEL_OFFSETS = {
    "Mistral Large 3":   (-0.10,  0.05, "right"),  # above, shifted left
    "Kimi K2":           (-0.10, -0.05, "right"),  # below, shifted left
    "GPT-OSS 120B":      (-0.10, -0.07, "right"),  # below, shifted left
}

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.subplots_adjust(top=0.90, bottom=0.10)
pt_colors = [GREEN if d > 0 else RED for d in piv2["delta"]]
ax.scatter(piv2["norag"], piv2["delta"], c=pt_colors, s=50, zorder=3,
           edgecolors="white", linewidths=0.4, alpha=0.9)
m_c, b_c = np.polyfit(piv2["norag"], piv2["delta"], 1)
x_r = np.linspace(piv2["norag"].min(), piv2["norag"].max(), 200)
ax.plot(x_r, m_c*x_r+b_c, color="#555", lw=1.3, ls="--", zorder=4)
ax.axhline(0, color="#888", lw=0.9, zorder=2)
ax.axhspan(piv2["delta"].min()-0.05, 0, color=RED, alpha=0.04, zorder=1)

for _, row in piv2.iterrows():
    if row["model_name"] in LABEL_MODELS:
        if row["model_name"] in LABEL_OFFSETS:
            dx, dy, ha = LABEL_OFFSETS[row["model_name"]]
        elif row["delta"] > 0:
            dx, dy, ha = 0.02, 0.03, "left"
        else:
            dx, dy, ha = 0.02, -0.05, "left"
        ax.annotate(row["model_name"],
                    xy=(row["norag"], row["delta"]),
                    xytext=(row["norag"]+dx, row["delta"]+dy),
                    fontsize=8, color="#333", ha=ha,
                    arrowprops=dict(arrowstyle="-", color="#AAA", lw=0.6))

ax.text(0.97, 0.97,
        f"Pearson r = {r_corr:.2f}, P = {r_p:.3f}\nSpearman ρ = {rho_c:.2f}, P = {rho_p:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCC", alpha=0.9))
ax.set_xlabel("Baseline composite quality (No-RAG)", fontsize=FS)
ax.set_ylabel("RAG benefit (Δ composite quality; RAG − No-RAG)", fontsize=FS)
ax.set_title("Baseline model quality was inversely associated with RAG benefit",
             fontweight="bold", pad=8)
ax.xaxis.grid(True, color=LGRAY, lw=0.5, zorder=0); ax.set_axisbelow(True)
save(fig, "Fig4_competence_paradox")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: Grounding metrics (2×2)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5 …")

rag_model = df[df["RAG_MODEL"]==True].groupby("model_name").agg(
    correctness=("correctness","mean"),
    citation_density=("evidence_density_per_100t","mean"),
    retrieval_align=("retrieval_alignment_score","mean"),
    gmkr=("gmkr","mean"),
    rur_citations=("rur_total_citations","mean"),
).reset_index()

panel_cfg = [
    ("citation_density", "A", "Citation density",         "Citations per 100 output tokens", -0.6447, 3.63e-5, True),
    ("retrieval_align",  "B", "Retrieval alignment",      "Retrieval alignment score",        -0.3230, 0.0449,  False),
    ("gmkr",             "C", "GMKR",                     "GMKR",                             0.0349,  0.833,   False),
    ("rur_citations",    "D", "Retrieval utilization rate","Total citations",                  0.1384,  0.401,   False),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(wspace=0.38, hspace=0.42, top=0.96, bottom=0.08)

for ax, (col, lbl, ptitle, xlabel, r_val, p_val, fdr_sig) in zip(axes.flat, panel_cfg):
    sub = rag_model[["correctness", col]].dropna()
    lc  = RED if fdr_sig else "#AAAAAA"
    ax.scatter(sub[col], sub["correctness"], color=BLUE, s=40, alpha=0.85,
               edgecolors="white", linewidths=0.3, zorder=3)
    m_g, b_g = np.polyfit(sub[col], sub["correctness"], 1)
    x_g = np.linspace(sub[col].min(), sub[col].max(), 100)
    ax.plot(x_g, m_g*x_g+b_g, color=lc, lw=1.5, zorder=4)
    if fdr_sig:       ann = f"r = {r_val:.2f}\nFDR q < 0.001"
    elif p_val < 0.05: ann = f"r = {r_val:.2f}\nP = {p_val:.3f}, q = 0.13"
    else:              ann = f"r = {r_val:.2f}\nP = {p_val:.2f}"
    ax.text(0.95, 0.95, ann, transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCC", alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel("Correctness (mean)", fontsize=FS)
    ax.set_title(ptitle, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=FS)
    ax.xaxis.grid(True, color=LGRAY, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.text(-0.14, 1.10, lbl, transform=ax.transAxes, fontsize=14, fontweight="bold")

save(fig, "Fig5_grounding_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: Judge reliability + judge-specific effects
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6 …")

judges = ["Claude-Sonnet-4.5", "GPT-5.2", "Gemini-3-Pro"]
short  = ["Sonnet-4.5", "GPT-5.2", "Gemini-3-Pro"]

pivot_j = df.pivot_table(
    index=["model_name","item_id","RAG_MODEL"],
    columns="judge_model", values="composite").reset_index()
pivot_j.columns.name = None

corr_matrix = np.zeros((3, 3))
for i, j1 in enumerate(judges):
    for j, j2 in enumerate(judges):
        if i == j:
            corr_matrix[i, j] = 1.0
        elif i < j:
            valid = pivot_j[[j1, j2]].dropna()
            r_val, _ = stats.pearsonr(valid[j1], valid[j2])
            corr_matrix[i, j] = corr_matrix[j, i] = r_val

jdf = df.groupby(["judge_model","RAG_MODEL"])["composite"].mean().reset_index()
judge_deltas = {
    jn: (jdf[(jdf["judge_model"]==jn) & (jdf["RAG_MODEL"]==True)]["composite"].values[0] -
         jdf[(jdf["judge_model"]==jn) & (jdf["RAG_MODEL"]==False)]["composite"].values[0])
    for jn in judges
}

fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
fig.subplots_adjust(wspace=0.44, top=0.88, bottom=0.10)

# Panel A — heatmap
ax = axes[0]
im = ax.imshow(corr_matrix, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=11, color="black" if corr_matrix[i,j] < 0.85 else "white")
ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=FS)
ax.set_yticks(range(3)); ax.set_yticklabels(short, fontsize=FS)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson r", fontsize=FS)
cbar.ax.tick_params(labelsize=FS)
ph(ax, "A", "Inter-judge agreement")

# Panel B — bars with white text inside
ax = axes[1]
j_labels = ["Sonnet-4.5", "GPT-5.2", "Gemini-3-Pro"]
deltas   = [judge_deltas[j] for j in judges]
bars = ax.bar(j_labels, deltas, color=[RED,RED,RED], width=0.5, edgecolor="none")
ax.axhline(0, color="#333", lw=0.9, zorder=5)
for bar, val in zip(bars, deltas):
    ax.text(bar.get_x()+bar.get_width()/2, val/2, f"{val:.3f}",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
ax.set_ylabel("Δ composite quality (RAG − No-RAG)", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.tick_params(axis="y", labelsize=FS)
ax.yaxis.grid(True, color=LGRAY, lw=0.5, zorder=0); ax.set_axisbelow(True)
ph(ax, "B", "Judge-specific RAG effect estimates")
save(fig, "Fig6_judge_reliability")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Error taxonomy
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7 …")

cat_rename = {
    "Critical omission":           "Omissions or missed differentials",
    "Ignores explicit case facts":  "Ignoring explicit case facts",
    "Dosing / quantitative":        "Dosing / quantitative error",
    "Drug interaction":             "Drug-interaction error",
    "Incorrect/unsafe management":  "Incorrect or unsafe management",
    "Source misattribution":        "Source misattribution",
    "Outdated guidance":            "Outdated guidance",
    "Other / unclassified":         "Other / unclassified",
}
t4p = t4.copy()
t4p["category"]   = t4p["category"].map(cat_rename).fillna(t4p["category"])
t4p["delta_rate"] = t4p["rate_RAG"] - t4p["rate_NoRAG"]
t4p = t4p.sort_values("delta_rate", ascending=False).reset_index(drop=True)
def err_stars(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""

fig, ax = plt.subplots(figsize=(7.5, max(6, len(t4p)*0.70)))
fig.subplots_adjust(top=0.88, bottom=0.08, left=0.30, right=0.93)

for i, row in t4p.iterrows():
    n_r = row["rate_NoRAG"]; r_r = row["rate_RAG"]
    color = RED if row["direction"] == "RAG increases" else GREEN
    ax.annotate("", xy=(r_r, i), xytext=(n_r, i),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=12), zorder=2)
    ax.scatter(n_r, i, s=65, color="white", edgecolors=BLUE, linewidths=2.0, zorder=4)
    ax.scatter(r_r, i, s=60, color=color, zorder=5, edgecolors="white", linewidths=0.5)
    s = err_stars(row["p"])
    if s: ax.text(max(n_r, r_r)+0.3, i, s, va="center", ha="left", fontsize=9, color="#444")

ax.set_yticks(np.arange(len(t4p)))
ax.set_yticklabels(t4p["category"].values, fontsize=FS)
ax.set_xlabel("Major-error rate per 100 judged generations", fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.set_xlim(0, 30)   # extended for legend clearance
ax.xaxis.grid(True, color=LGRAY, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title("RAG redistributes clinical errors: it helps lookup, impairs synthesis",
             fontweight="bold", pad=8)

legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor=BLUE, markeredgewidth=2.0, markersize=8, label="No-RAG (open)"),
    Line2D([0],[0], marker=">", color=RED,   lw=1.5, markerfacecolor=RED,   markersize=7, label="RAG — increase"),
    Line2D([0],[0], marker=">", color=GREEN, lw=1.5, markerfacecolor=GREEN, markersize=7, label="RAG — decrease"),
]
leg = ax.legend(handles=legend_els, frameon=True, facecolor="white", edgecolor="#AAAAAA",
                framealpha=1.0, borderpad=0.6, fontsize=9,
                loc="upper right", handlelength=2.0, handletextpad=0.6)
leg.get_frame().set_linewidth(0.8)
save(fig, "Fig7_error_redistribution")

print("\nAll 7 figures saved to", OUT)
