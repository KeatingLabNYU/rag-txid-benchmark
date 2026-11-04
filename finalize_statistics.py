#!/usr/bin/env python3
"""Reproduce revision statistics from the supplied raw CSVs. Python 3.10+, numpy/pandas."""
import hashlib
import itertools
import json
from math import exp, lgamma, log
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('RAG_PROJECT_DIR', Path.cwd())).resolve()  # project folder; set RAG_PROJECT_DIR or run from it
OUT = ROOT / 'revision_statistics'
DIMENSIONS = ['correctness', 'completeness', 'reasoning', 'relevance', 'clarity', 'consistency']
LLM_PATH = ROOT / 'data/all_models_rag_non_rag_judged.csv'
HUMAN_PATH = ROOT / 'data/expert_evaluations_linked_long.csv'
# Rating exports from the two evaluators added during peer review. These input files are not
# publicly released; file names and labels are generic. Released ratings use E1-E5.
NEW_PATHS = [(ROOT / 'added_evaluator_1_ratings.csv', 'Added evaluator 1'),
             (ROOT / 'added_evaluator_2_ratings.csv', 'Added evaluator 2')]
RETRIEVAL_PATH = ROOT / 'data/retrieval_quality_classification.csv'

def beta_cf(a, b, x):
    # Continued fraction for the regularized incomplete beta function.
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny: d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 1001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-13: break
    else:
        raise ArithmeticError('Incomplete beta failed to converge')
    return h

def ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * beta_cf(a, b, x) / a
    return 1 - bt * beta_cf(b, a, 1 - x) / b

def t_p_two_sided(t_value, df):
    return ibeta(df / 2, 0.5, df / (df + t_value * t_value))

def t_critical(df):
    low, high = 0.0, 100.0
    for _ in range(100):
        middle = (low + high) / 2
        if t_p_two_sided(middle, df) > 0.05: low = middle
        else: high = middle
    return (low + high) / 2

def cluster_covariance(scores, bread, keys):
    s = pd.DataFrame(scores).groupby(keys, sort=False).sum().to_numpy()
    clusters, observations, coefficients = len(s), len(scores), scores.shape[1]
    if clusters < 2: raise ValueError('At least two clusters needed')
    correction = clusters / (clusters - 1) * (observations - 1) / (observations - coefficients)
    return bread @ (s.T @ s) @ bread * correction

def two_way_ols(data, outcome):
    # Unadjusted output-level condition contrast, as described in submitted Methods.
    if len(data) != len(data[['model_name', 'item_id', 'RAG_MODEL']].drop_duplicates()):
        raise ValueError('Expected one row per output')
    y = data[outcome].to_numpy(float)
    treatment = data['RAG_MODEL'].astype(int).to_numpy()
    x = np.column_stack([np.ones(len(data)), treatment])
    bread = np.linalg.inv(x.T @ x)
    coef = bread @ x.T @ y
    scores = x * (y - x @ coef)[:, None]
    model = data['model_name'].reset_index(drop=True).astype(str)
    case = data['item_id'].reset_index(drop=True).astype(str)
    cov = (cluster_covariance(scores, bread, model)
           + cluster_covariance(scores, bread, case)
           - cluster_covariance(scores, bread, model + '|' + case))
    se = float(np.sqrt(max(cov[1, 1], 0)))
    df = min(model.nunique(), case.nunique()) - 1
    estimate = float(coef[1]); critical = t_critical(df)
    return {'estimate': estimate, 'se': se, 'df': df,
            'ci_low': estimate - critical * se, 'ci_high': estimate + critical * se,
            'p_value': t_p_two_sided(abs(estimate / se), df) if se else 0.0,
            'n_outputs': len(data), 'n_models': model.nunique(), 'n_cases': case.nunique(),
            'n_rag_outputs': int(treatment.sum()), 'n_no_rag_outputs': int(len(data)-treatment.sum()),
            'mean_rag': float(y[treatment == 1].mean()),
            'mean_no_rag': float(y[treatment == 0].mean())}

def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p); q = np.empty(len(p)); running = 1.0
    for pos in range(len(p)-1, -1, -1):
        idx = order[pos]
        running = min(running, p[idx] * len(p) / (pos+1))
        q[idx] = running
    return q

def human_panel(data, panel):
    cases = sorted(data.case_code.unique())
    n_evaluators = data.evaluator_name.nunique()
    if len(cases) != 10 or len(data) != 20 * n_evaluators:
        raise ValueError('Expected ten matched cases, two modes and complete evaluator panel')
    rows = []
    for outcome in ['accuracy', 'usefulness']:
        by_case = data.groupby(['case_code', 'mode'])[outcome].mean().unstack()
        delta = (by_case['RAG'] - by_case['No RAG']).loc[cases].to_numpy(float)
        # Enumerate all sign assignments at the independent case-pair level.
        signs = np.array(list(itertools.product([-1, 1], repeat=10)), dtype=float)
        permuted = np.abs(signs @ delta)
        p_exact = float(np.mean(permuted >= abs(delta.sum()) - 1e-12))
        mean = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(10)); crit = t_critical(9)
        rows.append({'panel': panel, 'outcome': outcome, 'n_evaluators': n_evaluators,
                     'n_independent_pairs': 10, 'n_ratings_per_mode': 10 * n_evaluators,
                     'mean_rag': float(data.loc[data['mode']=='RAG', outcome].mean()),
                     'mean_no_rag': float(data.loc[data['mode']=='No RAG', outcome].mean()),
                     'mean_difference': mean, 'case_level_se': se,
                     'case_t_ci_low': mean - crit*se, 'case_t_ci_high': mean + crit*se,
                     'exact_case_signflip_p': p_exact,
                     'ci_method': 't interval on ten case-averaged differences; df=9',
                     'test_method': 'two-sided exact case-level sign flip; 1024 assignments'})
    case_pairs = data.pivot_table(index=['evaluator_name','case_code'], columns='mode',
                                  values=['accuracy','usefulness','major_error'], aggfunc='first')
    case_pairs.columns = [f'{field}_{mode.replace(" ", "_")}' for field, mode in case_pairs.columns]
    case_pairs = case_pairs.reset_index()
    for endpoint in ['accuracy', 'usefulness', 'major_error']:
        case_pairs[f'{endpoint}_difference'] = (case_pairs[f'{endpoint}_RAG'].astype(float)
                                                 - case_pairs[f'{endpoint}_No_RAG'].astype(float))
    case_pairs['panel'] = panel
    return rows, case_pairs

def main():
    llm = pd.read_csv(LLM_PATH)
    expected = {'model_name','item_id','judge_model','RAG_MODEL','answer','has_major_error',*DIMENSIONS}
    if not expected.issubset(llm): raise ValueError('LLM columns missing')
    if len(llm) != 4911 or llm.model_name.nunique() != 39 or llm.item_id.nunique() != 21:
        raise ValueError('Unexpected automated dataset dimensions')
    llm['quality'] = llm[DIMENSIONS].mean(axis=1)
    keys = ['model_name','item_id','RAG_MODEL']
    count = llm.groupby(keys, dropna=False).size()
    if len(count) != 1637 or not count.eq(3).all():
        raise ValueError('Unexpected output/judge record structure')
    output = llm.groupby(keys, as_index=False)[['quality',*DIMENSIONS,'has_major_error']].mean()
    retrieval = pd.read_csv(RETRIEVAL_PATH)
    if set(retrieval.case_id) != set(llm.item_id): raise ValueError('Retrieval case mismatch')
    good = set(retrieval.loc[retrieval.retrieval_category=='Good retrieval','case_id'])
    excluded = set(retrieval.loc[retrieval.retrieval_category.isin(
        ['Near-zero similarity','Coverage failure','Corpus artifact','Deduplication failure']),'case_id'])
    eleven = set(retrieval.case_id) - excluded
    if len(good) != 7 or len(eleven) != 11 or not good.issubset(eleven):
        raise ValueError('Retrieval subset definitions invalid')
    scenarios = {'all_models_all_cases': output,
                 'good_retrieval_7_cases': output[output.item_id.isin(good)],
                 'exclude_severe_retrieval_failure_11_cases': output[output.item_id.isin(eleven)],
                 'bedrock_only': output[output.model_name.isin(llm.loc[llm.model_source.str.contains('Bedrock',case=False),'model_name'])],
                 'acute_A_cases': output[output.item_id.str.startswith('A-')],
                 'surveillance_S_cases': output[output.item_id.str.startswith('S-')]}
    endpoints = ['quality','correctness','has_major_error']
    results = []
    for label, subset in scenarios.items():
        for endpoint in endpoints:
            row = two_way_ols(subset, endpoint)
            row.update(analysis=label, outcome=endpoint,
                       method='judge-averaged output-level OLS; two-way model/case clustered covariance',
                       scale='1-5 points' if endpoint!='has_major_error' else 'judge-flag proportion risk difference')
            results.append(row)
    pd.DataFrame(results).to_csv(OUT/'automated_endpoint_results.csv',index=False,float_format='%.12g')

    dims = []
    for endpoint in DIMENSIONS:
        row = two_way_ols(output, endpoint)
        row['dimension'] = endpoint; dims.append(row)
    for row, q in zip(dims, bh_adjust([r['p_value'] for r in dims])):row['bh_q_six_dimensions'] = q
    pd.DataFrame(dims).to_csv(OUT/'automated_dimension_results.csv',index=False,float_format='%.12g')

    judge_rows = []
    for judge, subset in llm.groupby('judge_model'):
        # One rating per output for each judge; cluster by model/case.
        for endpoint in ['quality','correctness','has_major_error']:
            row = two_way_ols(subset, endpoint)
            row.update(judge=judge, outcome=endpoint)
            judge_rows.append(row)
    pd.DataFrame(judge_rows).to_csv(OUT/'judge_specific_results.csv',index=False,float_format='%.12g')

    original = pd.read_csv(HUMAN_PATH)
    original['major_error'] = original.major_error.astype(str).str.lower().eq('true')
    mapping = original[['case_code','side','model_name','mode']].drop_duplicates()
    if len(mapping)!=20 or mapping.duplicated(['case_code','side']).any():
        raise ValueError('Original A/B condition mapping is not unique')
    new_rows = []
    for path,name in NEW_PATHS:
        source = pd.read_csv(path)
        if len(source)!=10 or set(source.case_code)!=set(mapping.case_code):
            raise ValueError(f'{path.name}: incomplete ten-case evaluation')
        for _,r in source.iterrows():
            for side in ['a','b']:
                new_rows.append({'evaluator_name':name,'evaluation_date':r.evaluation_date,
                    'case_code':r.case_code,'case_title':r.case_title,
                    'self_rated_expertise':r.self_rated_expertise,
                    'preference':r.preference,'justification':r.justification,
                    'side':side.upper(),'accuracy':r[f'response_{side}_accuracy'],
                    'usefulness':r[f'response_{side}_usefulness'],
                    'major_error':r[f'response_{side}_major_error']=='Yes',
                    'comments':r[f'response_{side}_comments'], 'raw_source':path.name})
    new = pd.DataFrame(new_rows).merge(mapping,on=['case_code','side'],how='left',validate='many_to_one')
    if new[['model_name','mode']].isna().any().any() or len(new)!=40:
        raise ValueError('New expert A/B mapping failed')
    original['raw_source'] = HUMAN_PATH.name
    all_human = pd.concat([original,new],ignore_index=True)
    all_human = all_human[['evaluator_name','evaluation_date','case_code','case_title','model_name','side','mode',
                           'accuracy','usefulness','major_error','preference','self_rated_expertise',
                           'comments','justification','raw_source']]
    if len(all_human)!=100 or all_human.evaluator_name.nunique()!=5:
        raise ValueError('Five-expert panel dimensions incorrect')
    if all_human.duplicated(['evaluator_name','case_code','side']).any():
        raise ValueError('Duplicate evaluator-case-side row')
    all_human.to_csv(OUT/'all_five_expert_ratings_linked.csv',index=False)
    panels = [('original_three',original),('two_id_specialists',new),('combined_five',all_human)]
    human_rows=[];paired=[];descriptive=[];case_rows=[]
    for panel,data in panels:
        rr,pp=human_panel(data,panel)
        human_rows.extend(rr);paired.append(pp)
        for case,group in pp.groupby('case_code'):
            case_rows.append({'panel':panel,'case_code':case,'n_evaluators':len(group),
                              'accuracy_difference_mean':group.accuracy_difference.mean(),
                              'usefulness_difference_mean':group.usefulness_difference.mean(),
                              'major_error_flag_difference_sum':group.major_error_difference.sum()})
        for mode, group in data.groupby('mode'):
            descriptive.append({'panel':panel,'mode':mode,'n_ratings':len(group),
                                'accuracy_mean':group.accuracy.mean(),
                                'usefulness_mean':group.usefulness.mean(),
                                'major_error_flags':int(group.major_error.sum()),
                                'major_error_flag_rate':group.major_error.mean()})
    pd.DataFrame(human_rows).to_csv(OUT/'human_case_level_results.csv',index=False,float_format='%.12g')
    pd.concat(paired,ignore_index=True).to_csv(OUT/'human_evaluator_case_differences.csv',index=False)
    pd.DataFrame(case_rows).to_csv(OUT/'human_case_mean_differences.csv',index=False,float_format='%.12g')
    pd.DataFrame(descriptive).to_csv(OUT/'human_descriptive_results.csv',index=False,float_format='%.12g')

    # Preference is descriptive because evaluators repeatedly rated the same ten pairs.
    pref = all_human.drop_duplicates(['evaluator_name','case_code'])[['evaluator_name','case_code','preference']]
    side_mode = mapping.rename(columns={'mode':'preferred_mode'})
    pref = pref.merge(side_mode,left_on=['case_code','preference'],right_on=['case_code','side'],how='left')
    pref['preferred_mode'] = pref.preferred_mode.fillna('Tie')
    pref.to_csv(OUT/'human_preferences_linked.csv',index=False)

    # Missingness sensitivities: blank scored output versus absent output.
    blank_keys = llm.loc[llm.answer.isna(),keys].drop_duplicates()
    if len(blank_keys)!=1:
        raise ValueError('Expected one blank-answer output, separately from absent Cogito output')
    blank_tuple=tuple(blank_keys.iloc[0]); blank_mask=np.ones(len(output),dtype=bool)
    for column,value in zip(keys,blank_tuple):blank_mask &= output[column].eq(value)
    blank_sensitivity = [dict(analysis='including_scored_blank_answer',outcome=o,**two_way_ols(output,o))
                         for o in endpoints]
    no_blank=output.loc[~blank_mask]
    blank_sensitivity += [dict(analysis='excluding_scored_blank_answer',outcome=o,**two_way_ols(no_blank,o))
                          for o in endpoints]
    pd.DataFrame(blank_sensitivity).to_csv(OUT/'blank_answer_sensitivity.csv',index=False,float_format='%.12g')

    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [LLM_PATH,HUMAN_PATH,RETRIEVAL_PATH,*[p for p,_ in NEW_PATHS]]}
    meta={'input_sha256':hashes,'llm_judge_records':len(llm),'llm_unique_outputs':len(output),
          'llm_rag_outputs':int(output.RAG_MODEL.sum()),'llm_no_rag_outputs':int((~output.RAG_MODEL).sum()),
          'missing_model_case_condition':'Cogito 70B, A-01, No-RAG (no output/ratings)',
          'scored_blank_answer':dict(zip(keys,map(str,blank_tuple))),
          'retrieval_subsets':{'good_7':sorted(good),'severe_failure_excluded_11':sorted(eleven),
                               'excluded_10':sorted(excluded)},
          'human_new_mapping':'Author confirmed both new experts saw same A/B answers and order as original panel.',
          'human_inferential_unit':'ten case-model answer pairs; evaluator ratings averaged within pair',
          'limitations':['No protocol registration or dated analysis plan was provided.',
                         'Two-way clustered intervals use the smaller cluster count minus one for t reference.',
                         'Exact sign flip assumes case-level signs are exchangeable.',
                         'New expert data are post-review and do not add independent case pairs.',
                         'The analysis does not validate medical errors or causally isolate retrieval.']}
    (OUT/'analysis_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Saved revision statistics to',OUT)
    for r in results[:3]:
        print(r['outcome'],round(r['estimate'],4),
              (round(r['ci_low'],4),round(r['ci_high'],4)),r['p_value'])
    for r in human_rows[-2:]:
        print('combined',r['outcome'],round(r['mean_difference'],3),
              (round(r['case_t_ci_low'],3),round(r['case_t_ci_high'],3)),r['exact_case_signflip_p'])

if __name__ == '__main__':
    main()
