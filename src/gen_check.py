"""Check letter-probability scoring against greedy generated answers.

For a random sample of Analysis I items (unrotated order), the model generates an answer
with greedy decoding; the first option letter in the output is compared with the argmax
of the letter probabilities stored in results/<model>/<condition>.jsonl.

Usage: python gen_check.py --model qwen [--n 200] [--protocol final|noprefill|prefix_only]
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch

from common import LETTERS, build_prompt, read_jsonl
from score import PROTOCOLS, Scorer


def first_letter(text, k):
    m = re.search(rf"\b([{LETTERS[:k]}])\b", text)
    return LETTERS.index(m.group(1)) if m else None


@torch.no_grad()
def generate(scorer, prompts, langs, batch_size=16, max_new_tokens=16):
    """Greedy continuation of the same prompt (including the answer prefix) used for scoring."""
    tok = scorer.tok
    tok.padding_side = "left"
    outputs = []
    for i in range(0, len(prompts), batch_size):
        texts = [scorer.chat_text(p, lang) for p, lang in zip(prompts[i : i + batch_size], langs[i : i + batch_size])]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = scorer.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=scorer.pad_id)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", default="data/items.jsonl")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="final")
    args = parser.parse_args()
    if args.protocol != "final":
        args.results = os.path.join(args.results, args.protocol)

    items = [i for i in read_jsonl(args.items) if "a1" in i["sets"]]
    sample = random.Random(args.seed).sample(items, args.n)
    scorer = Scorer(args.model, 8192, args.protocol)
    report = {"model": args.model, "n": args.n}
    examples = []
    for condition in ("full", "choices_only"):
        probs = {r["uid"]: r["rotations"][0] for r in read_jsonl(os.path.join(args.results, args.model, f"{condition}.jsonl"))}
        outputs = generate(scorer, [build_prompt(item, condition, 0) for item in sample], [item["lang"] for item in sample])
        agree, parsed = 0, 0
        for item, text in zip(sample, outputs):
            k = len(item["options"])
            gen = first_letter(text, k)
            pick = int(np.argmax(probs[item["uid"]]))
            parsed += gen is not None
            agree += gen == pick
            examples.append({"condition": condition, "uid": item["uid"], "output": text, "generated": gen, "letter_prob": pick})
        report[condition] = {"parsed": parsed / args.n, "agreement": agree / args.n}
        print(condition, report[condition])
    out = os.path.join(args.results, args.model, "gen_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "examples": examples}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
