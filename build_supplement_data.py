"""Prepare typed matrices for the new supplementary workbooks."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(os.environ.get('RAG_PROJECT_DIR', Path.cwd())).resolve()
OUT=ROOT/'revision_submission'
STAT=ROOT/'revision_statistics'

def read_rows(path):
    return list(openpyxl.load_workbook(path,read_only=True,data_only=True).active.values)

auto=pd.read_csv(STAT/'automated_endpoint_results.csv')
dim=pd.read_csv(STAT/'automated_dimension_results.csv')
human=pd.read_csv(STAT/'human_case_level_results.csv')
desc=pd.read_csv(STAT/'human_descriptive_results.csv')

s1a=[['Analysis','Endpoint','RAG mean','No-RAG mean','Difference','95% CI low','95% CI high','P value','Units','Note']]
for scenario in ['all_models_all_cases','bedrock_only','acute_A_cases','surveillance_S_cases']:
    for _,r in auto[auto.analysis==scenario].iterrows():
        s1a.append([scenario,r.outcome,float(r.mean_rag),float(r.mean_no_rag),float(r.estimate),
                    float(r.ci_low),float(r.ci_high),float(r.p_value),r.scale,
                    f"{int(r.n_outputs)} outputs; {int(r.n_models)} models; {int(r.n_cases)} cases; model/case clustered"])
s1d=[['Dimension','RAG mean','No-RAG mean','Difference','95% CI low','95% CI high','P value','BH q (six tests)']]
for _,r in dim.iterrows():
    s1d.append([r.dimension,float(r.mean_rag),float(r.mean_no_rag),float(r.estimate),float(r.ci_low),
                float(r.ci_high),float(r.p_value),float(r.bh_q_six_dimensions)])
s1h=[['Panel','Outcome','Evaluators','Independent case pairs','RAG mean','No-RAG mean','Difference',
      '95% CI low','95% CI high','Exact sign-flip P','Note']]
for _,r in human.iterrows():
    s1h.append([r.panel,r.outcome,int(r.n_evaluators),int(r.n_independent_pairs),float(r.mean_rag),
                float(r.mean_no_rag),float(r.mean_difference),float(r.case_t_ci_low),
                float(r.case_t_ci_high),float(r.exact_case_signflip_p),
                'Two-sided exact test over ten case-average differences; post-review addition is exploratory'])
s1r=[['Subset','Endpoint','Cases','RAG mean','No-RAG mean','Difference','95% CI low','95% CI high','P value','Note']]
for scenario in ['good_retrieval_7_cases','exclude_severe_retrieval_failure_11_cases']:
    for _,r in auto[auto.analysis==scenario].iterrows():
        s1r.append([scenario,r.outcome,int(r.n_cases),float(r.mean_rag),float(r.mean_no_rag),float(r.estimate),
                    float(r.ci_low),float(r.ci_high),float(r.p_value),'Exploratory document-level subset; model/case clustered'])
s1p=[['Panel','Configuration','Ratings','Accuracy mean','Usefulness mean','Major-error flags','Flag rate']]
for _,r in desc.iterrows():
    s1p.append([r.panel,r['mode'],int(r.n_ratings),float(r.accuracy_mean),float(r.usefulness_mean),
                int(r.major_error_flags),float(r.major_error_flag_rate)])

raw=pd.read_csv(ROOT/'data/all_models_rag_non_rag_judged.csv')
dims=['correctness','completeness','reasoning','relevance','clarity','consistency']
raw['quality']=raw[dims].mean(axis=1)
out=raw.groupby(['model_name','item_id','RAG_MODEL'],as_index=False)[['quality','correctness','gmkr',
    'retrieval_alignment_score','evidence_density_per_100t','rur_distinct_docs','rur_total_citations']].mean()
rag=out[out.RAG_MODEL].groupby('model_name',as_index=False).mean(numeric_only=True)
s2=[['Metric','Outcome','Pearson r across models','Models','Interpretation']]
metrics=[('gmkr','General knowledge ratio'),('retrieval_alignment_score','Retrieval alignment'),
         ('evidence_density_per_100t','Citation density'),('rur_distinct_docs','Distinct cited documents'),
         ('rur_total_citations','Total citations')]
for metric,label in metrics:
    for outcome in ['correctness','quality']:
        subset=rag[[metric,outcome]].dropna()
        corr=float(np.corrcoef(subset[metric],subset[outcome])[0,1])
        s2.append([label,outcome,corr,len(subset),'Exploratory model-level association; not individual-answer prediction'])

old_s3=read_rows(ROOT/'S3_Table.xlsx')
s3=[['Automatic rationale category','No-RAG judge records','RAG judge records',
     'No-RAG rate per 100 ratings','RAG rate per 100 ratings','Coding limitation']]
for r in old_s3[1:]:
    s3.append([r[0],int(r[1]),int(r[2]),float(r[3]),float(r[4]),
               'Priority-ordered keyword assignment; not independent clinical adjudication'])

old_s4=read_rows(ROOT/'S4_Table.xlsx')
audit=pd.read_csv(ROOT/'api_retrieval_audit/audit_run/case_summary.csv').set_index('case_id')
s4=[list(old_s4[0])+['AI-assessed explicit passage coverage','Passage-audit limitation']]
for r in old_s4[1:]:
    coverage={'all':'all explicit requests','partial':'some explicit requests','none':'no explicit requests'}.get(
        audit.loc[r[0],'overall_coverage'],str(audit.loc[r[0],'overall_coverage']))
    s4.append(list(r)+[coverage,'Retrospective single-model AI label; not clinical sufficiency'])

payload={'S1':{'Automated':s1a,'Dimensions':s1d,'Human':s1h,'Retrieval':s1r,'Human flags':s1p},
         'S2':{'Correlations':s2},'S3':{'Rationale categories':s3},'S4':{'Retrieval cases':s4}}
(OUT/'supplement_matrices.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print('Prepared',[(k,list(v)) for k,v in payload.items()])
