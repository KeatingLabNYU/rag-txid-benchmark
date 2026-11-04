"""Recover the manifested text locally from an authorized document cache or export.

No API calls, retrieval reruns, arbitrary pickle globals, or public text uploads.
Every manifested candidate must match its UTF-8 SHA-256 before export succeeds.
"""
import argparse
import csv
import hashlib
import html
import json
import pickle
from pathlib import Path


class Document:
    pass


class CacheReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('haystack.dataclasses.document', 'Document'):
            return Document
        if module in ('numpy._core.multiarray', 'numpy.core.multiarray') and name == '_reconstruct':
            import numpy as np
            return np._core.multiarray._reconstruct
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            import numpy as np
            return getattr(np, name)
        raise pickle.UnpicklingError(f'Unsupported cache class: {module}.{name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--cache', type=Path, help='Authorized cached_documents_sapbert.pkl (requires NumPy)')
    group.add_argument('--passages-json', type=Path, help='Authorized all_recovered_passages.json export')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='Local private output directory')
    p.add_argument('--case', help='Optional case ID, e.g. S-07')
    args = p.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding='utf-8')))
    if args.case:
        rows = [r for r in rows if r['case_id'] == args.case]
        if not rows:
            p.error('Case not found in manifest')
    if args.cache:
        with args.cache.open('rb') as f:
            docs = CacheReader(f).load()
        texts = {d.id: d.content for d in docs}
    else:
        records = json.loads(args.passages_json.read_text())
        texts = {m['chunk_id']:m['full_passage'] for r in records for m in r['matches']}
    recovered = []
    for row in rows:
        text = texts.get(row['chunk_id'])
        if text is None or hashlib.sha256(text.encode('utf-8')).hexdigest() != row['passage_sha256']:
            raise SystemExit(f"Missing or mismatched text: {row['case_id']} slot {row['retrieval_slot']} candidate {row['candidate']}; no outputs written")
        recovered.append(dict(row, full_passage=text))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'verified_passages.json').write_text(json.dumps(recovered, ensure_ascii=False, indent=2)+'\n')
    blocks = ['<!doctype html><meta charset="utf-8"><title>Retrieved passage inspection</title>',
              '<style>body{max-width:1000px;margin:3em auto;padding:0 1em;font:17px/1.55 sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1em}article{border-top:1px solid #aaa;margin-top:2em}code{overflow-wrap:anywhere}</style>',
              '<h1>Retrieved passage inspection</h1><p>Local verified text, including extraction artifacts. This is not clinical validation. Alternative candidates are unresolved originals, not additional retrieval slots. Do not upload copyrighted text without permission.</p>']
    for r in recovered:
        esc = lambda k: html.escape(str(r[k]))
        blocks.append(f"<article><h2>{esc('case_id')} · slot {esc('retrieval_slot')} · candidate {esc('candidate')}</h2><p>{esc('source_title')}</p><p>Status: {esc('status')}. {esc('recovery_note')}</p><p>Chunk: <code>{esc('chunk_id')}</code><br>Text SHA-256: <code>{esc('passage_sha256')}</code></p><pre>{esc('full_passage')}</pre></article>")
    (args.output_dir/'verified_passages.html').write_text('\n'.join(blocks),encoding='utf-8')
    print(f'Verified and exported {len(recovered)} candidate passages to {args.output_dir}')


if __name__ == '__main__':
    main()
