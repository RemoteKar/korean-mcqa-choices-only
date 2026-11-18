"""Randomly sampled ChoicesOnly examples, successes and failures alike.

For each benchmark group, samples items uniformly at random (seed 0) and prints the options,
the gold answer, the model's rotation-averaged ChoicesOnly preference, and the longest option.

Usage: python examples.py --model qwen [--per-group 4]
"""
import argparse
import json
import os
import random

import numpy as np

from analyze import option_preference
from common import LETTERS, read_jsonl

GROUPS = {
    "KMMLU": lambda i: i["source"] == "kmmlu" and "a1" in i["sets"],
    "CLIcK exam": lambda i: i.get("click_origin") == "exam",
    "CLIcK textbook": lambda i: i.get("click_origin") == "textbook",
    "Global-MMLU ko": lambda i: i["source"] == "gmmlu_ko",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", default="data/items.jsonl")
    parser.add_argument("--results", default="results")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = read_jsonl(args.items)
    co = {r["uid"]: r["rotations"] for r in read_jsonl(os.path.join(args.results, args.model, "choices_only.jsonl"))}
    rng = random.Random(args.seed)
    out = []
    for name, keep in GROUPS.items():
        for item in rng.sample([i for i in items if keep(i)], args.per_group):
            pref = option_preference(co[item["uid"]])
            pick = int(np.argmax(pref))
            longest = int(np.argmax([len(str(o)) for o in item["options"]]))
            out.append({
                "group": name, "uid": item["uid"], "options": item["options"],
                "gold": LETTERS[item["gold"]], "model_pick": LETTERS[pick],
                "model_pref": [round(float(p), 3) for p in pref], "longest": LETTERS[longest],
                "correct": pick == item["gold"],
            })
            print(f"[{name}] {item['uid']}  gold={LETTERS[item['gold']]} pick={LETTERS[pick]} longest={LETTERS[longest]}")
            for j, o in enumerate(item["options"]):
                print(f"    {LETTERS[j]}. {o}   ({pref[j]:.2f})")
    path = os.path.join(args.results, args.model, "examples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
