"""Write the paper's LaTeX tables from results/tables/*.csv.

Usage: python make_tables.py [--results results] [--out results/latex]
Each table body is a complete tabular environment, included in the paper with \\input.
"""
import argparse
import csv
import json
import os
import re

MODELS = [("qwen", "Qwen"), ("exaone", "EXAONE")]


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x, digits=1):
    v = float(x)
    s = f"{abs(v):.{digits}f}"
    return f"$-${s}" if v < 0 and s.strip("0.") else s


def parse(cell):
    m = re.match(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", cell)
    return [float(g) for g in m.groups()]


def point(cell):
    return num(parse(cell)[0])


def ci(cell):
    m, lo, hi = parse(cell)
    return f"{num(m)} {{\\scriptsize[{num(lo)}, {num(hi)}]}}"


def pick(rows, **kw):
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def write(out, name, lines):
    with open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def table_a1(tables, out):
    a1, a3 = read(f"{tables}/a1.csv"), read(f"{tables}/a3.csv")
    lines = [r"\begin{tabular}{@{}lrrrrlrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Baselines} & & \multicolumn{5}{c}{Model accuracy and solvable fraction} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){7-11}",
             r"Benchmark & $n$ & Chance & Maj. & Long. & Model & \full{} & \textsc{CO} & $s_{\textsc{co}}$ & \textsc{OA} & \textsc{CO} fixed \\",
             r"\midrule"]
    for bench in ["KMMLU", "CLIcK (all)", "CLIcK exam", "CLIcK textbook"]:
        for j, (key, label) in enumerate(MODELS):
            r = pick(a1, model=key, benchmark=bench)
            rule = pick(a3, model=key, benchmark=bench)
            mr = lambda v: f"\\multirow{{2}}{{*}}{{{v}}}"
            left = (" & ".join([mr(bench), mr(f"{int(r['n']):,}"), mr(f"{100 * float(r['chance']):.1f}"),
                                mr(f"{100 * float(r['majority']):.1f}"), mr(point(rule['rule_longest']))])
                    if j == 0 else " & & & & ")
            lines.append(f"{left} & {label} & {point(r['full'])} & {point(r['co'])} & {ci(r['s_co'])} & "
                         f"{point(r['oa'])} & {point(r['co_unrot'])} \\\\")
        lines.append(r"\midrule" if bench != "CLIcK textbook" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write(out, "a1", lines)


def table_a2(tables, out):
    """Analysis II: both models side by side, with the differences below."""
    a2 = read(f"{tables}/a2.csv")
    d, gap = read(f"{tables}/a2_deltas.csv"), read(f"{tables}/polarity_gap.csv")
    names = [("native (KMMLU)", "Native (KMMLU)"), ("translated (Global-MMLU ko)", "Translated (GMMLU-ko)"),
             ("English source (Global-MMLU en)", "English source (GMMLU-en)")]
    lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Qwen} & \multicolumn{4}{c}{EXAONE} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"Source & \full{} & \textsc{CO} & $s_{\textsc{co}}$ & Share & \full{} & \textsc{CO} & $s_{\textsc{co}}$ & Share \\",
             r"\midrule"]
    for src, name in names:
        cells = []
        for key, _ in MODELS:
            r = pick(a2, model=key, source=src)
            cells += [point(r["full"]), point(r["co"]), ci(r["s_co"]), point(r["share"])]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    rows = [(r"$\Delta_{\mathrm{orig}}$", "d_orig", d),
            (r"$\Delta_{\mathrm{share}}$", "d_share", d),
            (r"$\Delta_{\mathrm{tr}}$ (paired)", "d_tr", d),
            (r"$\Delta_{\mathrm{orig}}$, positive only$^{\dagger}$", "d_orig_positive", gap)]
    for label, col, src in rows:
        vals = [ci(pick(src, model=key)[col]) for key, _ in MODELS]
        lines.append(f"{label} & \\multicolumn{{4}}{{c}}{{{vals[0]}}} & \\multicolumn{{4}}{{c}}{{{vals[1]}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "a2", lines)


def table_polarity(tables, out):
    pol = read(f"{tables}/polarity.csv")
    benches = ["KMMLU", "CLIcK exam", "CLIcK textbook", "KMMLU (A2 domains)", "Global-MMLU ko", "Global-MMLU en"]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Positive} & \multicolumn{2}{c}{Negated} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
             r"Benchmark & Neg.\ share & Qwen & EXAONE & Qwen & EXAONE \\", r"\midrule"]
    for b in benches:
        cells = []
        share = None
        for pol_name in ("positive", "negated"):
            for key, _ in MODELS:
                r = pick(pol, model=key, benchmark=b, polarity=pol_name)
                cells.append(ci(r["s_co"]))
                if pol_name == "negated":
                    share = float(r["share"])
        lines.append(f"{b} & {100 * share:.0f}\\% & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "polarity", lines)


def table_mink(tables, out):
    mk = read(f"{tables}/a1_mink.csv")
    benches = ["KMMLU", "CLIcK exam", "CLIcK textbook"]
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Benchmark & Model & Q1 & Q2 & Q3 & Q4 & Q5 & Q1--4 \\", r"\midrule"]
    for b in benches:
        for key, label in MODELS:
            cells = [point(pick(mk, model=key, benchmark=b, quintile=q)["s_co"]) for q in "12345"]
            cells.append(ci(pick(mk, model=key, benchmark=b, quintile="1-4")["s_co"]))
            lines.append(f"{b if key == 'qwen' else ''} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "mink", lines)


def table_rules(tables, out):
    a3 = read(f"{tables}/a3.csv")
    benches = ["KMMLU", "CLIcK exam", "CLIcK textbook", "Global-MMLU ko", "Global-MMLU en"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Rule accuracy} & \multicolumn{3}{c}{Agreement (Q / E)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Benchmark & Longest & Digits & Odd & Longest & Digits & Odd \\", r"\midrule"]
    for b in benches:
        q, e = pick(a3, model="qwen", benchmark=b), pick(a3, model="exaone", benchmark=b)
        acc = [point(q[f"rule_{r}"]) for r in ("longest", "most_digits", "odd_one_out")]
        agr = [f"{100 * float(q[f'agree_{r}']):.0f} / {100 * float(e[f'agree_{r}']):.0f}"
               for r in ("longest", "most_digits", "odd_one_out")]
        lines.append(f"{b} & " + " & ".join(acc + agr) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "rules", lines)


def table_domains(tables, out):
    dom = read(f"{tables}/a2_domains.csv")
    domains = sorted({r["domain"] for r in dom})
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Negated} & \multicolumn{2}{c}{Qwen $s_{\textsc{co}}$} & \multicolumn{2}{c}{EXAONE $s_{\textsc{co}}$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Domain & nat & tr & nat & tr & nat & tr \\", r"\midrule"]
    for d in domains:
        q, e = pick(dom, model="qwen", domain=d), pick(dom, model="exaone", domain=d)
        cells = [f"{100 * float(q['neg_nat']):.0f}\\%", f"{100 * float(q['neg_tr']):.0f}\\%"]
        cells += [num(100 * float(x)) for x in (q["s_nat"], q["s_tr"], e["s_nat"], e["s_tr"])]
        lines.append(f"{d.replace('-', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "domains", lines)


def table_protocols(results, out):
    """Generation agreement and key statistics under the three scoring protocols."""
    variants = [("noprefill", "No prefix"), ("prefix_only", "Prefix"), ("final", "Prefix + bold (final)")]
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Protocol & Model & Agree (\full) & Agree (\copt) & $s_{\textsc{co}}$ KMMLU & $\Delta_{\mathrm{orig}}$ \\",
             r"\midrule"]
    for name, label in variants:
        base = results if name == "final" else os.path.join(results, name)
        for key, mlabel in MODELS:
            gpath = os.path.join(base, key, "gen_check.json")
            if not os.path.exists(gpath):
                continue
            with open(gpath, encoding="utf-8") as f:
                rep = json.load(f)["report"]
            a1 = pick(read(os.path.join(base, "tables", "a1.csv")), model=key, benchmark="KMMLU")
            dl = pick(read(os.path.join(base, "tables", "a2_deltas.csv")), model=key)
            lines.append(f"{label} & {mlabel} & {100 * rep['full']['agreement']:.1f} & "
                         f"{100 * rep['choices_only']['agreement']:.1f} & {ci(a1['s_co'])} & {ci(dl['d_orig'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "protocols", lines)


def table_examples(results, out):
    """Index of the randomly sampled ChoicesOnly examples (examples.py, seed 0)."""
    with open(os.path.join(results, "qwen", "examples.json"), encoding="utf-8") as f:
        qwen = json.load(f)
    with open(os.path.join(results, "exaone", "examples.json"), encoding="utf-8") as f:
        exaone = {e["uid"]: e for e in json.load(f)}
    lines = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
             r"Group & Item & Polarity & Gold & Longest & Qwen & EXAONE \\", r"\midrule"]
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from analyze import is_negated
    from common import read_jsonl
    items = {i["uid"]: i for i in read_jsonl("data/items.jsonl")}
    for e in qwen:
        x = exaone[e["uid"]]
        uid = e["uid"].split("/", 1)[1].replace("_", r"\_")
        pol = "negated" if is_negated(items[e["uid"]]) else "positive"
        lines.append(f"{e['group']} & \\texttt{{{uid}}} & {pol} & {e['gold']} & {e['longest']} & "
                     f"{e['model_pick']} & {x['model_pick']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "examples", lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/latex")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tables = os.path.join(args.results, "tables")
    for fn in (table_a1, table_a2, table_polarity, table_mink, table_rules, table_domains):
        fn(tables, args.out)
    table_protocols(args.results, args.out)
    table_examples(args.results, args.out)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
