# Answering Without the Question — code

Code for the paper "Answering Without the Question: Do Korean Multiple-Choice Benchmarks Measure
Knowledge or Option Artifacts?". It tests whether items in KMMLU, CLIcK and Global-MMLU (Korean and
English) can be answered from their options alone, with two small open models. Everything is
inference only.

## Environment

- Tested on Windows 11, Python 3.10, one NVIDIA RTX 5070 Ti (16 GB), CUDA 12.8 wheels
- `torch==2.11.0+cu128` and `transformers==4.49.0`; the other packages are pinned in `requirements.txt`
  (torch is installed separately because its CUDA build comes from the PyTorch index)
- Models run in bfloat16: `Qwen/Qwen2.5-3B-Instruct` and `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`.
  EXAONE needs `trust_remote_code=True` and is pinned to revision `e949c91` (December 2024), because
  later revisions of its model code require transformers v5.
- Datasets and models are downloaded from the Hugging Face Hub on first use (no login needed).

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Reproduce

Run from this folder. The random seed is 0 everywhere (item sampling, generation-check sample,
bootstrap with B = 2,000, random examples).

| Step | Command | Output | Time (RTX 5070 Ti) |
|---|---|---|---|
| 1. Build the 5,195 items | `python src/build_items.py` | `data/items.jsonl` | ~2 min |
| 2. Score each model | `python src/score.py --model qwen` and `--model exaone` | `results/<model>/*.jsonl` | ~30 and ~20 min |
| 3. Letter vs. generation check | `python src/gen_check.py --model qwen` and `--model exaone` | `results/<model>/gen_check.json` | ~2 min each |
| 4. All result tables (CSV) | `python src/analyze.py --models qwen exaone` | `results/tables/*.csv` | <1 min |
| 5. Random examples | `python src/examples.py --model qwen` and `--model exaone` | `results/<model>/examples.json` | <1 min |
| 6. LaTeX tables and Figure 1 | `python src/make_tables.py` and `python src/make_figure.py` | `results/latex/` | <1 min |

Step 6 also needs the protocol variants below (for the protocol table), so run them first if you
want that table.

### Paper tables

| Paper | File | Made by |
|---|---|---|
| Table 1 (Analysis I) | `results/latex/a1.tex` | `results/tables/a1.csv`, `a3.csv` |
| Table 2 (Analysis II and differences) | `results/latex/a2.tex` | `results/tables/a2.csv`, `a2_deltas.csv`, `polarity_gap.csv` |
| Figure 1 (polarity) | `results/latex/polarity.pdf` | `results/tables/polarity.csv` |
| Table 3 (examples) | written by hand from `results/<model>/examples.json` | |
| Appendix tables | `domains.tex`, `protocols.tex`, `polarity.tex`, `mink.tex`, `rules.tex`, `examples.tex` | |

### Scoring protocol variants (appendix)

The paper's tables use the final protocol. Two earlier variants are kept because one conclusion
changed between them (see the paper's appendix on protocol revisions):

```bash
python src/score.py --model qwen   --protocol noprefill --tasks full,choices_only
python src/score.py --model exaone --protocol noprefill --tasks full,choices_only
python src/score.py --model exaone --protocol prefix_only --tasks full,choices_only
python src/gen_check.py --model qwen   --protocol noprefill     # likewise for the other runs
cp results/qwen/{option_alone,mink}.jsonl results/noprefill/qwen/      # these tasks do not depend on the protocol
cp results/exaone/{option_alone,mink}.jsonl results/noprefill/exaone/
python src/analyze.py --results results/noprefill --models qwen exaone
python src/analyze.py --results results/prefix_only --models exaone
```

## Scoring protocol

Each prompt is a user turn; the assistant turn is opened with `정답:` (Korean items) or `Answer:`
(English items). The probability of letter L is
`P(L | prefix) + P(" **" | prefix) * P(L | prefix + " **")`, where L may be tokenised as `A` or ` A`,
renormalised over the item's k letters. The second term covers EXAONE, which opens its answers with
a markdown bold marker. Every item is scored under all k cyclic rotations of its options.

## Question polarity (exploratory)

`src/analyze.py` labels a question as negated with a regular expression on the last sentence of the
stem (`NEGATION` and `is_negated`). The rule was written after seeing the main results and has only
been spot-checked, not validated against manual labels.

## Notes

- Eight CLIcK ids occur twice with different items; `build_items.py` numbers the repeats
  (`#2`) so that every uid is unique.
- `data/` is not included in this repository because it contains benchmark text; step 1 rebuilds it.
