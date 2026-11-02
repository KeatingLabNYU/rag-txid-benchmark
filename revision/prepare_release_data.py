"""Create deidentified supporting data and source table for reproducible figures."""
from pathlib import Path
import csv
import shutil
import pandas as pd
import openpyxl

root = Path(__file__).resolve().parents[2]
submission = root / 'revision_submission'
data = submission / 'data'
data.mkdir(exist_ok=True)
stats = root / 'revision_statistics'

ratings = pd.read_csv(stats / 'all_five_expert_ratings_linked.csv')
names = list(pd.unique(ratings.evaluator_name))
assert len(names) == 5 and len(ratings) == 100
anonymous = {name: f'E{i+1}' for i, name in enumerate(names)}
ratings.insert(0, 'evaluator_id', ratings.evaluator_name.map(anonymous))
rating_columns = ['evaluator_id','case_code','side','model_name','mode','accuracy',
                  'usefulness','major_error','preference']
ratings[rating_columns].to_csv(data / 'expert_ratings_deidentified.csv', index=False)

paired = pd.read_csv(stats / 'human_evaluator_case_differences.csv')
paired = paired.loc[paired.panel == 'combined_five'].copy()
paired.insert(0, 'evaluator_id', paired.evaluator_name.map(anonymous))
assert len(paired) == 50 and paired.evaluator_id.notna().all()
paired.drop(columns=['evaluator_name','panel']).to_csv(
    data / 'expert_case_differences_deidentified.csv', index=False)
case = pd.read_csv(stats / 'human_case_mean_differences.csv')
case.loc[case.panel == 'combined_five'].drop(columns='panel').to_csv(
    data / 'case_mean_differences.csv', index=False)

shutil.copyfile(root / 'rag-txid-benchmark-main' / 'complex_prompts.json',
                data / 'acute_case_prompts.json')
shutil.copyfile(root / 'rag-txid-benchmark-main' / 'simple_prompts.json',
                data / 'surveillance_case_prompts.json')

sheet = openpyxl.load_workbook(root / 'S3_Table.xlsx', read_only=True,
                               data_only=True).active
with (submission / 's3_source.csv').open('w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows(sheet.values)
assert 'evaluator_name' not in pd.read_csv(data / 'expert_ratings_deidentified.csv').columns
print('Prepared deidentified data, case prompts and figure source table.')
