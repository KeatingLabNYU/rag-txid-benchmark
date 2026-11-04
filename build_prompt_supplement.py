"""Extract the exact prompt templates currently present in the supplied source scripts."""
import os
from pathlib import Path

ROOT=Path(os.environ.get('RAG_PROJECT_DIR', Path.cwd())).resolve()
SRC=Path(__file__).resolve().parent  # generation and judge scripts in this repository
OUT=ROOT/'revision_submission'/'S6_Prompt_templates_from_supplied_scripts.md'

def template(file,marker):
    source=(SRC/file).read_text(encoding='utf-8')
    start=source.index(marker)+len(marker)
    end=source.index('"""',start)
    return source[start:end].strip('\n')

sections=[
('Bedrock No RAG system prompt','run_bedrock_model_without_RAG.py','CLINICIAN_SYSTEM_PROMPT = """'),
('Bedrock RAG user template','run_bedrock_model_with_RAG_metrics.py','        template = """'),
('Ollama No RAG user template','run_ollama_model_NO_RAG.py','        template = """'),
('Ollama RAG user template','run_ollama_model_with_RAG_metrics.py','        template = """'),
('LLM judge user template','judge_all_models_multidim_csv.py','        return f"""'),
]
parts=['# Prompt templates from supplied source scripts',
'These templates are copied exactly from the local scripts supplied for this revision. They make the configuration differences inspectable. The scripts alone do not establish which commit and settings produced every historical answer; final authors should verify them against original run records before calling them the exact executed prompts.',
'The 21 case questions are supplied separately in `data/acute_case_prompts.json` and `data/surveillance_case_prompts.json`. The RAG templates insert recovered documents as `doc.content`; the generated CSV stores only short source previews.',
'Source defaults include Bedrock manager temperature 0.1 and a 4096-token default, with model-specific overrides and token clamping; Ollama generation uses temperature 0.1 and `num_predict=4096` in its script. Judge calls set temperature 0. Confirm actual per-model run settings from logs.']
for label,file,marker in sections:
    parts += [f'## {label}',f'Source: `{file}`','```text',template(file,marker),'```']
OUT.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
print(OUT)
