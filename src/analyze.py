"""Compute all tables from the score files.

Usage: python analyze.py --models qwen exaone [--results results] [--tag ""]
Writes CSV and LaTeX tables to <results>/tables/ and prints a summary.
"""
import argparse
import csv
import os
import re
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from common import read_jsonl

B = 2000
SEED = 0

# Questions asking for the false / non-matching option ("옳지 않은 것은?", "Which is NOT ...").
# Exploratory: defined after seeing the Analysis I results; spot-checked by hand (see README).
NEGATION = {
    "ko": re.compile(r"(않은|않는|아닌|틀린|옳지|거리가 먼|없는|해당하지|잘못된|부적절|적절하지|맞지 않)"),
    "en": re.compile(r"\b(NOT|not|EXCEPT|except|false|incorrect|least)\b"),
}


# ---------------------------------------------------------------- per-item scores

def question_stem(question):
    """Last sentence of the question, i.e. the part that says what to choose."""
    pieces = [x.strip() for x in re.split(r"[.!\n]", question.strip().rstrip("?")) if x.strip()]
    return pieces[-1] if pieces else question


def is_negated(item):
    stem = question_stem(item["question"])
    if item["lang"] == "ko":
        # Korean states what to choose at the end ("... 옳지 않은 것은"): require the negation to be
        # followed by at most 12 characters, so that negations inside the content
        # ("아이를 낳지 않은 며느리를 어떻게 부르는가") are not counted.
        return bool(re.search(NEGATION["ko"].pattern + r".{0,12}$", stem))
    return bool(NEGATION[item["lang"]].search(stem))


def rotation_accuracy(rotations, gold):
    """Fraction of rotations whose argmax position holds the gold option."""
    k = len(rotations)
    hits = [int(np.argmax(p)) == (gold - r) % k for r, p in enumerate(rotations)]
    return float(np.mean(hits))


def unrotated_correct(rotations, gold):
    return float(int(np.argmax(rotations[0])) == gold)


def option_preference(rotations):
    """Rotation-averaged probability of each original option (letter bias cancels)."""
    k = len(rotations)
    pref = np.zeros(k)
    for r, p in enumerate(rotations):
        for pos, prob in enumerate(p):
            pref[(pos + r) % k] += prob
    return pref / k


def tie_share(values, gold, pick_max=True):
    """Expected accuracy of argmax (or argmin) when ties are broken at random."""
    values = np.asarray(values, dtype=float)
    best = values.max() if pick_max else values.min()
    winners = np.flatnonzero(np.isclose(values, best))
    return float(gold in winners) / len(winners), winners


def char_ngrams(text, n=3):
    text = re.sub(r"\s+", " ", str(text))
    return {text[i : i + n] for i in range(max(1, len(text) - n + 1))}


def rule_scores(options):
    lengths = [len(str(o)) for o in options]
    digits = [len(re.findall(r"\d", str(o))) for o in options]
    grams = [char_ngrams(o) for o in options]
    sims = []
    for i, g in enumerate(grams):
        others = [len(g & h) / max(1, len(g | h)) for j, h in enumerate(grams) if j != i]
        sims.append(float(np.mean(others)))
    return {
        "longest": (lengths, True),
        "most_digits": (digits, True),
        "odd_one_out": (sims, False),
    }


def build_table(items, scores):
    rows = []
    for item in items:
        uid, gold, k = item["uid"], item["gold"], len(item["options"])
        if uid not in scores["full"]:
            continue
        full, co = scores["full"][uid], scores["choices_only"][uid]
        row = {
            "uid": uid, "gold": gold, "source": item["source"], "sets": item["sets"], "domain": item.get("domain"),
            "pair_id": item.get("pair_id"), "click_origin": item.get("click_origin"), "k": k,
            "chance": 1 / k,
            "full": rotation_accuracy(full, gold),
            "co": rotation_accuracy(co, gold),
            "full_unrot": unrotated_correct(full, gold),
            "co_unrot": unrotated_correct(co, gold),
            "negated": is_negated(item),
        }
        if uid in scores.get("option_alone", {}):
            row["oa"], _ = tie_share(scores["option_alone"][uid], gold)
        if uid in scores.get("mink", {}):
            row["mink"] = scores["mink"][uid]
        model_pick = int(np.argmax(option_preference(co)))
        for name, (values, pick_max) in rule_scores(item["options"]).items():
            acc, winners = tie_share(values, gold, pick_max)
            row[f"rule_{name}"] = acc
            row[f"agree_{name}"] = float(model_pick in winners) / len(winners)
        rows.append(row)
    return rows


def solvable(acc, chance):
    return (acc - chance) / (1 - chance)


# ---------------------------------------------------------------- bootstrap

def boot_mean(values, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    stats = values[idx].mean(axis=1)
    return values.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def boot_ratio(num, den, rng):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    stats = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return num.mean() / den.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def boot_diff(a, b, rng, paired=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        return boot_mean(a - b, rng)
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    stats = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return a.mean() - b.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def fmt(stat, scale=100, digits=1):
    m, lo, hi = stat
    return f"{m * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


# ---------------------------------------------------------------- analyses

def a1_groups(rows):
    return {
        "KMMLU": [r for r in rows if r["source"] == "kmmlu" and "a1" in r["sets"]],
        "CLIcK (all)": [r for r in rows if r["source"] == "click"],
        "CLIcK exam": [r for r in rows if r.get("click_origin") == "exam"],
        "CLIcK textbook": [r for r in rows if r.get("click_origin") == "textbook"],
    }


def analysis_1(model, rows, rng):
    out = []
    for name, group in a1_groups(rows).items():
        if not group:
            continue
        chance = np.mean([r["chance"] for r in group])
        s_co = [solvable(r["co"], r["chance"]) for r in group]
        s_full = [solvable(r["full"], r["chance"]) for r in group]
        acc_co = boot_mean([r["co"] for r in group], rng)
        golds = Counter(r["gold"] for r in group)
        out.append({
            "model": model, "benchmark": name, "n": len(group), "chance": chance,
            # fixed-order majority baseline: always answer the most frequent gold letter
            "majority": max(golds.values()) / len(group),
            "full": boot_mean([r["full"] for r in group], rng),
            "co": acc_co,
            "oa": boot_mean([r["oa"] for r in group], rng) if "oa" in group[0] else None,
            "s_full": boot_mean(s_full, rng),
            "s_co": boot_mean(s_co, rng),
            "full_unrot": boot_mean([r["full_unrot"] for r in group], rng),
            "co_unrot": boot_mean([r["co_unrot"] for r in group], rng),
            "h1_supported": acc_co[1] > chance,
        })
    return out


def analysis_1_mink(model, rows, rng):
    out = []
    for name, group in a1_groups(rows).items():
        if not group or "mink" not in group[0]:
            continue
        cuts = np.percentile([r["mink"] for r in group], [20, 40, 60, 80])
        for r in group:
            r["quintile"] = int(np.searchsorted(cuts, r["mink"], side="right")) + 1  # 5 = most likely seen
        for q in range(1, 6):
            sub = [r for r in group if r["quintile"] == q]
            out.append({"model": model, "benchmark": name, "quintile": str(q), "n": len(sub),
                        "s_co": boot_mean([solvable(r["co"], r["chance"]) for r in sub], rng),
                        # share of GPT-4-generated CLIcK items in this quintile (point estimate)
                        "textbook_share": np.mean([r.get("click_origin") == "textbook" for r in sub])})
        sub = [r for r in group if r["quintile"] < 5]
        out.append({"model": model, "benchmark": name, "quintile": "1-4", "n": len(sub),
                    "s_co": boot_mean([solvable(r["co"], r["chance"]) for r in sub], rng)})
    return out


def analysis_2(model, rows, rng):
    groups = {
        "native (KMMLU)": [r for r in rows if r["source"] == "kmmlu" and "a2" in r["sets"]],
        "translated (Global-MMLU ko)": [r for r in rows if r["source"] == "gmmlu_ko"],
        "English source (Global-MMLU en)": [r for r in rows if r["source"] == "gmmlu_en"],
    }
    if not all(groups.values()):
        return [], {}
    out = []
    s = {}
    for name, group in groups.items():
        s_co = [solvable(r["co"], r["chance"]) for r in group]
        s_full = [solvable(r["full"], r["chance"]) for r in group]
        s[name] = (s_co, s_full)
        out.append({"model": model, "source": name, "n": len(group),
                    "full": boot_mean([r["full"] for r in group], rng),
                    "co": boot_mean([r["co"] for r in group], rng),
                    "s_full": boot_mean(s_full, rng), "s_co": boot_mean(s_co, rng),
                    "share": boot_ratio(s_co, s_full, rng),
                    # point estimates only (no extra bootstrap draws)
                    "co_unrot": np.mean([r["co_unrot"] for r in group]),
                    "majority": max(Counter(r["gold"] for r in group).values()) / len(group)})
    nat, tr, en = groups.values()
    ko_by_pair = {r["pair_id"]: r for r in tr}
    en_by_pair = {r["pair_id"]: r for r in en}
    pairs = sorted(set(ko_by_pair) & set(en_by_pair))
    deltas = {
        "model": model,
        "d_orig": boot_diff(s["translated (Global-MMLU ko)"][0], s["native (KMMLU)"][0], rng),
        "d_share": None,
        "d_tr": boot_diff([solvable(ko_by_pair[p]["co"], 0.25) for p in pairs],
                          [solvable(en_by_pair[p]["co"], 0.25) for p in pairs], rng, paired=True),
        "n_pairs": len(pairs),
    }
    # shortcut-share difference, bootstrapped jointly
    a_co, a_full = (np.asarray(x) for x in s["translated (Global-MMLU ko)"])
    b_co, b_full = (np.asarray(x) for x in s["native (KMMLU)"])
    ia = rng.integers(0, len(a_co), size=(B, len(a_co)))
    ib = rng.integers(0, len(b_co), size=(B, len(b_co)))
    stats = a_co[ia].mean(1) / a_full[ia].mean(1) - b_co[ib].mean(1) / b_full[ib].mean(1)
    point = a_co.mean() / a_full.mean() - b_co.mean() / b_full.mean()
    deltas["d_share"] = (point, np.percentile(stats, 2.5), np.percentile(stats, 97.5))
    deltas["h2_supported"] = deltas["d_orig"][1] > 0 and deltas["d_share"][0] > 0
    return out, deltas


def analysis_2_domains(model, rows, rng):
    out = []
    domains = sorted({r["domain"] for r in rows if r["domain"]})
    for d in domains:
        nat_rows = [r for r in rows if r["source"] == "kmmlu" and r["domain"] == d and "a2" in r["sets"]]
        tr_rows = [r for r in rows if r["source"] == "gmmlu_ko" and r["domain"] == d]
        nat = [solvable(r["co"], r["chance"]) for r in nat_rows]
        tr = [solvable(r["co"], r["chance"]) for r in tr_rows]
        if nat and tr:
            out.append({"model": model, "domain": d, "n": len(nat),
                        "s_nat": np.mean(nat), "s_tr": np.mean(tr), "diff": boot_diff(tr, nat, rng),
                        "neg_nat": np.mean([r["negated"] for r in nat_rows]),
                        "neg_tr": np.mean([r["negated"] for r in tr_rows])})
    return out


def analysis_polarity(model, rows, rng):
    """Exploratory: ChoicesOnly solvability split by question polarity, and the origin gap on
    positive questions only."""
    groups = dict(a1_groups(rows))
    groups["KMMLU (A2 domains)"] = [r for r in rows if r["source"] == "kmmlu" and "a2" in r["sets"]]
    groups["Global-MMLU ko"] = [r for r in rows if r["source"] == "gmmlu_ko"]
    groups["Global-MMLU en"] = [r for r in rows if r["source"] == "gmmlu_en"]
    out = []
    for name, group in groups.items():
        for negated in (False, True):
            sub = [r for r in group if r["negated"] == negated]
            if not sub:
                continue
            out.append({"model": model, "benchmark": name, "polarity": "negated" if negated else "positive",
                        "n": len(sub), "share": len(sub) / len(group),
                        "s_co": boot_mean([solvable(r["co"], r["chance"]) for r in sub], rng),
                        # point estimate only, so that the bootstrap stream (and all CIs) is unchanged
                        "s_full": np.mean([solvable(r["full"], r["chance"]) for r in sub])})
    nat = [solvable(r["co"], r["chance"]) for r in groups["KMMLU (A2 domains)"] if not r["negated"]]
    tr = [solvable(r["co"], r["chance"]) for r in groups["Global-MMLU ko"] if not r["negated"]]
    gap = {"model": model, "d_orig_positive": boot_diff(tr, nat, rng), "n_nat": len(nat), "n_tr": len(tr)}
    # Across the ten domains: does a larger negated-share gap go with a larger origin gap?
    xs, ys = [], []
    for d in sorted({r["domain"] for r in rows if r["domain"]}):
        nat_d = [r for r in groups["KMMLU (A2 domains)"] if r["domain"] == d]
        tr_d = [r for r in groups["Global-MMLU ko"] if r["domain"] == d]
        xs.append(np.mean([r["negated"] for r in nat_d]) - np.mean([r["negated"] for r in tr_d]))
        ys.append(np.mean([solvable(r["co"], r["chance"]) for r in tr_d])
                  - np.mean([solvable(r["co"], r["chance"]) for r in nat_d]))
    rho = spearmanr(xs, ys)
    gap["domain_spearman"], gap["domain_spearman_p"] = float(rho.statistic), float(rho.pvalue)
    return out, gap


def analysis_3(model, rows, rng):
    out = []
    groups = dict(a1_groups(rows))
    groups["Global-MMLU ko"] = [r for r in rows if r["source"] == "gmmlu_ko"]
    groups["Global-MMLU en"] = [r for r in rows if r["source"] == "gmmlu_en"]
    for name, group in groups.items():
        if not group:
            continue
        row = {"model": model, "benchmark": name, "n": len(group),
               "chance": np.mean([r["chance"] for r in group]),
               "co": boot_mean([r["co"] for r in group], rng)}
        for rule in ("longest", "most_digits", "odd_one_out"):
            row[f"rule_{rule}"] = boot_mean([r[f"rule_{rule}"] for r in group], rng)
            row[f"agree_{rule}"] = np.mean([r[f"agree_{rule}"] for r in group])
        out.append(row)
    return out


# ---------------------------------------------------------------- output

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([fmt(v) if isinstance(v, tuple) else v for v in (r.get(k) for k in keys)])


def load_scores(results, model):
    scores = {}
    for task in ("full", "choices_only", "option_alone", "mink"):
        path = os.path.join(results, model, f"{task}.jsonl")
        if not os.path.exists(path):
            continue
        rows = read_jsonl(path)
        key = {"full": "rotations", "choices_only": "rotations", "option_alone": "scores", "mink": "mink"}[task]
        scores[task] = {r["uid"]: r[key] for r in rows}
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen", "exaone"])
    parser.add_argument("--items", default="data/items.jsonl")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    items = read_jsonl(args.items)
    tables = {k: [] for k in ("a1", "a1_mink", "a2", "a2_deltas", "a2_domains", "a3", "polarity", "polarity_gap")}
    item_rows = {}
    for model in args.models:
        rng = np.random.default_rng(SEED)
        rows = build_table(items, load_scores(args.results, model))
        item_rows[model] = rows
        tables["a1"] += analysis_1(model, rows, rng)
        tables["a1_mink"] += analysis_1_mink(model, rows, rng)
        a2, deltas = analysis_2(model, rows, rng)
        tables["a2"] += a2
        if deltas:
            tables["a2_deltas"].append(deltas)
        tables["a2_domains"] += analysis_2_domains(model, rows, rng)
        tables["a3"] += analysis_3(model, rows, rng)
        polarity, gap = analysis_polarity(model, rows, rng)
        tables["polarity"] += polarity
        tables["polarity_gap"].append(gap)

    out_dir = os.path.join(args.results, "tables")
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in tables.items():
        write_csv(os.path.join(out_dir, f"{name}.csv"), rows)
    for model, rows in item_rows.items():
        write_csv(os.path.join(out_dir, f"items_{model}.csv"),
                  [{k: v for k, v in r.items() if k != "sets"} for r in rows])

    for name, rows in tables.items():
        print(f"\n== {name}")
        for r in rows:
            print("  " + " | ".join(f"{k}={fmt(v) if isinstance(v, tuple) else (f'{v:.3f}' if isinstance(v, float) else v)}" for k, v in r.items()))


if __name__ == "__main__":
    main()
