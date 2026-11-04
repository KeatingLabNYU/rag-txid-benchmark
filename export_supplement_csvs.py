"""Export S1–S4 table matrices to portable CSV files without spreadsheet software."""
import os
from pathlib import Path
import csv
import json

submission = Path(os.environ.get('RAG_PROJECT_DIR', Path.cwd())).resolve() / 'revision_submission'
matrices = json.loads((submission / 'supplement_matrices.json').read_text(encoding='utf-8'))
out = submission / 'data' / 'tables_csv'
out.mkdir(parents=True, exist_ok=True)
for supplement, sheets in matrices.items():
    for name, rows in sheets.items():
        target = out / f'{supplement}_{name.lower().replace(" ", "_")}.csv'
        with target.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
print('Exported', sum(map(len, matrices.values())), 'tables to', out)
