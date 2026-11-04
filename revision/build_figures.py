"""Create separate journal figures from finalized analysis tables, in the visual style of the original submission."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _font in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'):
    if Path(_font).exists():
        font_manager.fontManager.addfont(_font)
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'revision_submission' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
STAT = ROOT / 'revision_statistics'
BLUE, RED, GREEN, GREY, LINE = '#2c6e9b', '#c0392b', '#1a8a5a', '#999999', '#dddddd'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8.5,
                     'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'savefig.facecolor': 'white', 'legend.fontsize': 8})


def save(fig, n):
    # PLOS: RGB, 300 dpi, at most 7.5 in (2,250 px) wide and 8.75 in (2,625 px) tall.
    path = OUT / f'Fig{n}.tif'
    fig.savefig(path, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    with Image.open(path) as im:
        rgb = im.convert('RGB')
    assert rgb.size[0] <= 2250 and rgb.size[1] <= 2625, (n, rgb.size)
    rgb.save(path, compression='tiff_lzw', dpi=(300, 300))


def panel(ax, letter):
    ax.text(-0.02, 1.13, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right', va='bottom')


def xgrid(ax):
    ax.grid(axis='x', color=LINE, linewidth=0.6); ax.set_axisbelow(True)


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


judge = pd.read_csv(ROOT / 'data/all_models_rag_non_rag_judged.csv')
dimensions = ['correctness', 'completeness', 'reasoning', 'relevance', 'clarity', 'consistency']
judge['quality'] = judge[dimensions].mean(axis=1)
output = judge.groupby(['model_name', 'item_id', 'RAG_MODEL'], as_index=False)[['quality', *dimensions,
    'gmkr', 'retrieval_alignment_score', 'evidence_density_per_100t', 'rur_distinct_docs', 'rur_total_citations']].mean()
dim_result = pd.read_csv(STAT / 'automated_dimension_results.csv').set_index('dimension')
human = pd.read_csv(STAT / 'human_case_level_results.csv')
human = human.loc[human.panel == 'combined_five'].set_index('outcome')
model = output.groupby(['model_name', 'RAG_MODEL']).quality.mean().unstack()
model['delta'] = model[True] - model[False]

# Figure 1: six scoring dimensions (dumbbells) and the case-level clinical panel.
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={'width_ratios': [1.25, 1], 'wspace': .45})
order = dim_result.loc[dimensions, 'estimate'].sort_values().index.tolist()
for i, d in enumerate(order):
    no, ra = dim_result.loc[d, 'mean_no_rag'], dim_result.loc[d, 'mean_rag']
    ax.plot([ra, no], [i, i], color=LINE, linewidth=2.5, zorder=1)
    ax.scatter(no, i, color=BLUE, s=38, zorder=2); ax.scatter(ra, i, color=RED, s=38, zorder=2)
    ax.text(max(no, ra) + .06, i, stars(dim_result.loc[d, 'bh_q_six_dimensions']), va='center', fontsize=8, color='#444444')
ax.set_yticks(range(6), [d.capitalize() for d in order]); ax.invert_yaxis(); ax.set_xlim(2.4, 5.1)
ax.set_xlabel('Mean score (1–5 scale)'); ax.set_title('LLM-judge ratings by configuration', loc='left'); xgrid(ax)
ax.scatter([], [], color=BLUE, s=30, label='No-RAG'); ax.scatter([], [], color=RED, s=30, label='RAG')
ax.legend(loc='upper right', frameon=True, edgecolor='#bbbbbb'); panel(ax, 'A')
for i, d in enumerate(['usefulness', 'accuracy']):
    no, ra = human.loc[d, 'mean_no_rag'], human.loc[d, 'mean_rag']
    bx.plot([ra, no], [i, i], color=LINE, linewidth=2.5, zorder=1)
    bx.scatter(no, i, color=BLUE, s=38, zorder=2); bx.scatter(ra, i, color=RED, s=38, zorder=2)
    bx.text(no + .07, i, stars(human.loc[d, 'exact_case_signflip_p']), va='center', fontsize=8, color='#444444')
    bx.text(no, i + .22, f'{no:.2f}', ha='center', va='top', fontsize=8, color=BLUE)
    bx.text(ra, i + .22, f'{ra:.2f}', ha='center', va='top', fontsize=8, color=RED)
bx.set_yticks([0, 1], ['Usefulness', 'Accuracy']); bx.set_ylim(1.6, -.6); bx.set_xlim(2.4, 5.1)
bx.set_xlabel('Mean rating (1–5 scale)'); bx.set_title('Expert ratings, ten selected pairs', loc='left'); xgrid(bx); panel(bx, 'B')
save(fig, 1)

# Figure 2: per-model composite quality difference, horizontal bars.
m2 = model.sort_values('delta', ascending=False)
fig, ax = plt.subplots(figsize=(6.8, 7.6))
yy = np.arange(len(m2))
ax.barh(yy, m2.delta, height=.65, color=[GREEN if v > 0 else RED for v in m2.delta])
for y, (name, v) in zip(yy, m2.delta.items()):
    if v > 0: ax.text(v + .02, y, name, va='center', fontsize=8, color=GREEN)
ax.axvline(0, color='#333333', linewidth=.8); ax.set_yticks(yy, m2.index, fontsize=8); ax.invert_yaxis()
ax.set_xlabel('RAG − No-RAG composite quality'); xgrid(ax)
ax.set_title('Model-level differences in composite quality', loc='left')
save(fig, 2)

# Figure 3: preferences (stacked bar) and descriptive human–judge relationship.
pref = pd.read_csv(STAT / 'human_preferences_linked.csv').preferred_mode.value_counts()
linked = pd.read_csv(STAT / 'all_five_expert_ratings_linked.csv')
linked['RAG_MODEL'] = linked['mode'].eq('RAG')
hm = linked.groupby(['case_code', 'model_name', 'RAG_MODEL'], as_index=False).accuracy.mean()
jm = output.rename(columns={'item_id': 'case_code'})[['case_code', 'model_name', 'RAG_MODEL', 'correctness']]
paired = hm.merge(jm, on=['case_code', 'model_name', 'RAG_MODEL'], validate='one_to_one')
assert len(paired) == 20
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={'wspace': .35})
left = 0
for key, color, label in [('RAG', RED, 'RAG preferred'), ('Tie', GREY, 'Tie'), ('No RAG', BLUE, 'No-RAG preferred')]:
    n = int(pref.get(key, 0))
    ax.barh(0, n, left=left, color=color, height=.5, label=label)
    ax.text(left + n / 2, 0, str(n), ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    left += n
ax.set_xlim(0, left); ax.set_ylim(-.6, .9); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Evaluator–case decisions (5 evaluators × 10 pairs)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1)
ax.set_title('Paired preferences', loc='left'); panel(ax, 'A')
for condition, color, label in [(False, BLUE, 'No-RAG'), (True, RED, 'RAG')]:
    part = paired[paired.RAG_MODEL == condition]
    bx.scatter(part.correctness, part.accuracy, color=color, label=label, s=34, alpha=.9, edgecolor='white', linewidth=.5)
bx.set(xlim=(.8, 4.6), ylim=(2, 4.2), xlabel='LLM-judged correctness (mean)', ylabel='Expert accuracy rating (mean)')
bx.grid(color=LINE, linewidth=.6); bx.set_axisbelow(True)
bx.legend(loc='lower right', frameon=True, edgecolor='#bbbbbb'); bx.set_title('Expert and LLM-judge scores, 20 answers', loc='left')
panel(bx, 'B')
save(fig, 3)

# Figure 4: baseline score and observed difference; descriptive only.
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.axhspan(model.delta.min() - .05, 0, color=RED, alpha=.04, zorder=0)
ax.axhline(0, color='#888888', linewidth=.8)
ax.scatter(model[False], model.delta, s=30, zorder=2, edgecolor='white', linewidth=.5,
           color=[GREEN if v > 0 else RED for v in model.delta])
slope, intercept = np.polyfit(model[False], model.delta, 1)
xs = np.linspace(model[False].min(), model[False].max(), 50)
ax.plot(xs, intercept + slope * xs, '--', color='#555555', linewidth=1)
labels = {'DeepSeek-R1 8B': (8, 6), 'Nemotron Nano 12B': (8, 6), 'Jamba 1.5 Mini': (8, -10), 'Jamba 1.5 Large': (8, -10),
          'Kimi K2': (-45, -12), 'Mistral Large 3': (-75, 10), 'GPT-OSS 120B': (-75, -14)}
for name, (dx, dy) in labels.items():
    if name in model.index:
        ax.annotate(name, (model.loc[name, False], model.loc[name, 'delta']), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8, color='#333333',
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', linewidth=.6))
r = np.corrcoef(model[False], model.delta)[0, 1]
ax.text(.98, .97, f'Pearson r = {r:.2f}\n(descriptive; difference contains baseline)', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc'))
ax.set(xlabel='Baseline composite quality (No-RAG)', ylabel='RAG − No-RAG composite quality')
ax.grid(axis='x', color=LINE, linewidth=.6); ax.set_axisbelow(True)
ax.set_title('Baseline quality and observed configuration difference', loc='left')
save(fig, 4)

# Figure 5: four exploratory model-level grounding metrics.
rag = output[output.RAG_MODEL].groupby('model_name', as_index=False).mean(numeric_only=True)
metrics = [('evidence_density_per_100t', 'Citation density', 'Citations per 100 output tokens'),
           ('retrieval_alignment_score', 'Retrieval alignment', 'Retrieval alignment score'),
           ('gmkr', 'GMKR', 'General medical knowledge ratio'),
           ('rur_total_citations', 'Total citations', 'Total citations per answer (model mean)')]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={'hspace': .55, 'wspace': .32})
for ax, letter, (column, title, xlabel) in zip(axes.flat, 'ABCD', metrics):
    part = rag[[column, 'correctness']].dropna()
    ax.scatter(part[column], part.correctness, color=BLUE, s=24, alpha=.9, edgecolor='white', linewidth=.4, zorder=2)
    b, a = np.polyfit(part[column], part.correctness, 1)
    xs = np.linspace(part[column].min(), part[column].max(), 50)
    ax.plot(xs, a + b * xs, color=GREY, linewidth=1.2, zorder=1)
    r = np.corrcoef(part[column], part.correctness)[0, 1]
    ax.text(.97, .95, f'r = {r:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc'))
    ax.set(xlabel=xlabel, ylabel='Correctness (mean)'); ax.set_title(title)
    xgrid(ax); panel(ax, letter)
save(fig, 5)

# Figure 6: inter-judge agreement and judge-specific output-level estimates.
wide = judge.pivot_table(index=['model_name', 'item_id', 'RAG_MODEL'], columns='judge_model', values='quality')
names = ['Claude-Sonnet-4.5', 'GPT-5.2', 'Gemini-3-Pro']
short = ['Sonnet-4.5', 'GPT-5.2', 'Gemini-3-Pro']
corr = wide[names].corr().to_numpy()
jr = pd.read_csv(STAT / 'judge_specific_results.csv')
jr = jr[jr.outcome == 'quality'].set_index('judge').loc[names]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={'width_ratios': [1, 1.1], 'wspace': .45})
im = ax.imshow(corr, cmap='Blues', vmin=.5, vmax=1, aspect='auto')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if corr[i, j] > .85 else 'black')
ax.set_xticks(range(3), short, fontsize=8); ax.set_yticks(range(3), short, fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04); cb.set_label('Pearson r'); cb.ax.tick_params(labelsize=7)
ax.set_title('Inter-judge score correlations', loc='left'); panel(ax, 'A')
xx = np.arange(3)
bx.errorbar(xx, jr.estimate, yerr=[jr.estimate - jr.ci_low, jr.ci_high - jr.estimate], fmt='o', color=RED,
            ecolor='#333333', markersize=6, capsize=3, linewidth=.9, zorder=3)
for x, v in zip(xx, jr.estimate):
    bx.text(x + .09, v, f'{v:.3f}', ha='left', va='center', fontsize=8, color='#333333', fontweight='bold')
bx.set_ylim(jr.ci_low.min() - .06, max(.08, jr.ci_high.max() + .03))
bx.axhline(0, color='#333333', linewidth=.8); bx.set_xticks(xx, short, fontsize=8); bx.set_xlim(-.5, 2.8)
bx.set_ylabel('Δ composite quality (RAG − No-RAG)')
bx.grid(axis='y', color=LINE, linewidth=.6); bx.set_axisbelow(True)
bx.set_title('Judge-specific estimates (95% CI)', loc='left'); panel(bx, 'B')
save(fig, 6)

# Figure 7: original S3 category counts as descriptive rates; no inferential marks.
S3SRC = Path(__file__).resolve().parent / 's3_source.csv'
if not S3SRC.exists():
    print('s3_source.csv not found; Figure 7 not regenerated'); sys.exit(0)
s3 = pd.read_csv(S3SRC)
s3['change'] = s3.rate_RAG - s3.rate_NoRAG
s3 = s3.sort_values('change')
fig, ax = plt.subplots(figsize=(6.9, 4.4))
for i, row in enumerate(s3.itertuples()):
    color = RED if row.change > 0 else GREEN
    if abs(row.change) > .15:
        ax.annotate('', xy=(row.rate_RAG, i), xytext=(row.rate_NoRAG, i),
                    arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.4, mutation_scale=9, shrinkA=4, shrinkB=3))
    ax.scatter(row.rate_RAG, i, color=color, s=30, zorder=3)
    ax.scatter(row.rate_NoRAG, i, facecolor='white', edgecolor=BLUE, linewidth=1.6, s=42, zorder=4)
ax.set_yticks(range(len(s3)), s3.category); ax.invert_yaxis(); ax.set_xlim(0, 22)
ax.set_xlabel('Judge records assigned the category (%)'); xgrid(ax)
ax.scatter([], [], facecolor='white', edgecolor=BLUE, linewidth=1.6, s=36, label='No-RAG')
ax.scatter([], [], color=RED, s=26, label='RAG, higher rate')
ax.scatter([], [], color=GREEN, s=26, label='RAG, lower rate')
ax.legend(loc='center right', frameon=True, edgecolor='#bbbbbb')
ax.set_title('Automatically coded judge-rationale categories', loc='left')
save(fig, 7)
print('Saved seven figures to', OUT)
