"""Second-round additions: judge same-developer sensitivity and the S-07 leflunomide unit count."""
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('RAG_PROJECT_DIR', Path.cwd())).resolve()
judge = pd.read_csv(ROOT / 'data/all_models_rag_non_rag_judged.csv')
dimensions = ['correctness', 'completeness', 'reasoning', 'relevance', 'clarity', 'consistency']
judge['quality'] = judge[dimensions].mean(axis=1)
judge['rag'] = judge.RAG_MODEL.astype(str).eq('True').astype(int)


def t_cdf(t, df):
    from math import lgamma
    x = np.linspace(0, abs(t), 20001)
    c = np.exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / np.sqrt(df * np.pi)
    area = np.trapezoid(c * (1 + x ** 2 / df) ** (-(df + 1) / 2), x)
    return 0.5 + np.sign(t) * area


def two_way_cluster(y, x, g1, g2):
    """OLS slope with two-way cluster-robust SE; t reference with min(clusters)-1 df."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)

    def vcov(groups):
        codes = pd.factorize(groups)[0]
        meat = np.zeros((2, 2))
        for k in np.unique(codes):
            s = (X[codes == k] * resid[codes == k, None]).sum(0)
            meat += np.outer(s, s)
        n, c = len(y), len(np.unique(codes))
        return c / (c - 1) * (n - 1) / (n - 2) * bread @ meat @ bread, c

    v1, c1 = vcov(g1)
    v2, c2 = vcov(g2)
    v12, _ = vcov(pd.Series(g1).astype(str) + '|' + pd.Series(g2).astype(str))
    se = np.sqrt((v1 + v2 - v12)[1, 1])
    df = min(c1, c2) - 1
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < 0.975 else (lo, mid)
    p = 2 * (1 - t_cdf(abs(beta[1] / se), df))
    return beta[1], beta[1] - mid * se, beta[1] + mid * se, p


# Judge same-developer sensitivity: drop each judge's ratings of its own developer's models.
family = {'Claude-Sonnet-4.5': 'claude', 'GPT-5.2': 'gpt-oss', 'Gemini-3-Pro': 'gemma'}
same = [family[j] in m.lower() for j, m in zip(judge.judge_model, judge.model_name)]
kept = judge[~np.array(same)]
out = kept.groupby(['model_name', 'item_id', 'rag'], as_index=False).quality.mean()
est = two_way_cluster(out.quality.values, out.rag.values, out.model_name.values, out.item_id.values)
print(f'Excluded same-developer judge records: {sum(same)} of {len(judge)}')
print('Quality difference excluding same-developer ratings: %.3f (95%% CI %.3f to %.3f; P=%.4f)' % est)

# S-07: model-written leflunomide targets in mg/mL, excluding the appended source previews.
answers = judge.drop_duplicates(['model_name', 'item_id', 'RAG_MODEL'])
s07 = answers[answers.item_id == 'S-07']
preview = r'prevention of BKVN remains an optimal therapeutic trough blood level of 50-100mg/ml\. Ma\.\.\.'
for flag, label in [(True, 'RAG'), (False, 'No-RAG')]:
    texts = s07[s07.RAG_MODEL == flag].answer.fillna('').map(lambda t: re.sub(preview, '', t))
    hits = texts.map(lambda t: bool(re.search(r'eflunomide', t, re.I)) and
                     bool(re.search(r'(trough|level|target)[^.\n]{0,60}\d\s*mg/m[lL]', t, re.I)))
    print(f'S-07 {label}: {hits.sum()} of {len(texts)} answers give a mg/mL leflunomide target')
