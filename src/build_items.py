"""Download the benchmarks and write the evaluation items to data/items.jsonl.

Analysis I  : KMMLU test, 40 items per subject (45 subjects) + all CLIcK items.
Analysis II : 10 subject domains shared by KMMLU and MMLU; 60 KMMLU items per domain
              (the first 40 are the Analysis I items) and 60 Global-MMLU items per domain,
              in Korean and in English (aligned by sample_id).
"""
import argparse
import os
import random
from collections import Counter

from datasets import load_dataset

from common import write_jsonl

KMMLU_SUBJECTS = [
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance", "Biology",
    "Chemical-Engineering", "Chemistry", "Civil-Engineering", "Computer-Science", "Construction",
    "Criminal-Law", "Ecology", "Economics", "Education", "Electrical-Engineering",
    "Electronics-Engineering", "Energy-Management", "Environmental-Science", "Fashion",
    "Food-Processing", "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
    "Information-Technology", "Interior-Architecture-and-Design", "Korean-History", "Law",
    "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering", "Marketing",
    "Materials-Engineering", "Math", "Mechanical-Engineering", "Nondestructive-Testing", "Patent",
    "Political-Science-and-Sociology", "Psychology", "Public-Safety",
    "Railway-and-Automotive-Engineering", "Real-Estate", "Refrigerating-Machinery", "Social-Welfare",
    "Taxation", "Telecommunications-and-Wireless-Technology",
]

# KMMLU subject -> MMLU subjects covering the same domain (Analysis II).
DOMAINS = {
    "Accounting": ["professional_accounting"],
    "Biology": ["college_biology", "high_school_biology"],
    "Chemistry": ["college_chemistry", "high_school_chemistry"],
    "Computer-Science": ["college_computer_science", "high_school_computer_science"],
    "Economics": ["econometrics", "high_school_macroeconomics", "high_school_microeconomics"],
    "Law": ["professional_law"],
    "Management": ["management"],
    "Marketing": ["marketing"],
    "Math": ["college_mathematics", "high_school_mathematics", "elementary_mathematics"],
    "Psychology": ["high_school_psychology", "professional_psychology"],
}

N_A1_PER_SUBJECT = 40
N_A2_PER_DOMAIN = 60


def kmmlu_items(rng):
    items = []
    for subject in KMMLU_SUBJECTS:
        ds = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        order = list(range(len(ds)))
        rng.shuffle(order)
        n = N_A2_PER_DOMAIN if subject in DOMAINS else N_A1_PER_SUBJECT
        for rank, idx in enumerate(order[:n]):
            row = ds[idx]
            sets = []
            if rank < N_A1_PER_SUBJECT:
                sets.append("a1")
            if subject in DOMAINS:
                sets.append("a2")
            items.append({
                "uid": f"kmmlu/{subject}/{idx}",
                "source": "kmmlu",
                "lang": "ko",
                "subject": subject,
                "domain": subject if subject in DOMAINS else None,
                "sets": sets,
                "paragraph": "",
                "question": row["question"],
                "options": [row["A"], row["B"], row["C"], row["D"]],
                "gold": int(row["answer"]) - 1,  # KMMLU answers are 1-indexed
            })
    return items


def click_items():
    ds = load_dataset("EunsuKim/CLIcK", split="train")
    items, skipped, seen = [], 0, Counter()
    for row in ds:
        choices = list(row["choices"])
        if choices.count(row["answer"]) != 1:
            skipped += 1
            continue
        # Eight CLIcK ids occur twice with different items; number the repeats to keep uids unique.
        seen[row["id"]] += 1
        suffix = "" if seen[row["id"]] == 1 else f"#{seen[row['id']]}"
        items.append({
            "uid": f"click/{row['id']}{suffix}",
            "source": "click",
            "lang": "ko",
            "subject": row["id"].rsplit("_", 1)[0],
            "click_origin": "textbook" if row["id"].startswith("KIIP") else "exam",
            "domain": None,
            "sets": ["a1"],
            "paragraph": row["paragraph"] or "",
            "question": row["question"],
            "options": choices,
            "gold": choices.index(row["answer"]),
        })
    print(f"CLIcK: kept {len(items)}, skipped {skipped} (answer not exactly one of the choices)")
    return items


def gmmlu_items(rng):
    ko = load_dataset("CohereLabs/Global-MMLU", "ko", split="test")
    en = load_dataset("CohereLabs/Global-MMLU", "en", split="test")
    en_by_id = {row["sample_id"]: row for row in en}
    items = []
    for domain, subjects in DOMAINS.items():
        pool = [row for row in ko if row["subject"] in subjects]
        pool.sort(key=lambda row: row["sample_id"])
        rng.shuffle(pool)
        for row in pool[:N_A2_PER_DOMAIN]:
            for lang, src in (("ko", row), ("en", en_by_id[row["sample_id"]])):
                items.append({
                    "uid": f"gmmlu_{lang}/{row['sample_id']}",
                    "pair_id": row["sample_id"],
                    "source": f"gmmlu_{lang}",
                    "lang": lang,
                    "subject": row["subject"],
                    "domain": domain,
                    "culture": row["cultural_sensitivity_label"],
                    "sets": ["a2"],
                    "paragraph": "",
                    "question": src["question"],
                    "options": [src["option_a"], src["option_b"], src["option_c"], src["option_d"]],
                    "gold": "ABCD".index(src["answer"]),
                })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/items.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = kmmlu_items(rng) + click_items() + gmmlu_items(rng)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_jsonl(args.out, items)

    print(f"wrote {len(items)} items to {args.out}")
    print("by source:", Counter(i["source"] for i in items))
    print("options per item:", Counter((i["source"], len(i["options"])) for i in items))
    print("gold position:", Counter((i["source"], i["gold"]) for i in items))
    print("CLIcK origin:", Counter(i.get("click_origin") for i in items if i["source"] == "click"))


if __name__ == "__main__":
    main()
