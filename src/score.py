"""Score every item with one model.

Writes results/<model>/<task>.jsonl for four tasks:
  full          letter probabilities with question + options, for every cyclic rotation
  choices_only  letter probabilities with options only, for every cyclic rotation
  option_alone  length-normalised log-likelihood of each option scored in isolation
  mink          Min-K% Prob (K=20%) of the full item text

Usage: python score.py --model qwen [--limit 300] [--tasks full,choices_only,option_alone,mink]
                       [--protocol final|noprefill|prefix_only]

--protocol selects how letter probabilities are read (full and choices_only only):
  final        assistant turn opened with "정답:"/"Answer:"; letters as "A" or " A"; plus the
               " **" bold-marker path (see run_letters). Used for all tables in the paper.
  noprefill    no answer prefix; letters as "A" only. Earlier variant, kept for the appendix.
  prefix_only  answer prefix; letters as "A" or " A"; no bold-marker path. Earlier variant.
Non-final protocols write to results/<protocol>/<model>/.
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import LETTERS, PROMPTS, build_prompt, item_text, read_jsonl

# (repo, pinned revision). EXAONE is pinned to its December 2024 code, which runs on
# transformers 4.49; later revisions of its remote code require transformers v5.
MODELS = {
    "qwen": ("Qwen/Qwen2.5-3B-Instruct", None),
    "exaone": ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "e949c91dec92095908d34e6b560af77dd0c993f8"),
}
MINK_RATIO = 0.2
BOLD = " **"
MAX_TOKENS = 2048


PROTOCOLS = ("final", "noprefill", "prefix_only")


class Scorer:
    def __init__(self, name, token_budget, protocol="final"):
        assert protocol in PROTOCOLS, protocol
        self.protocol = protocol
        path, revision = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, revision=revision, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        self.base = getattr(self.model, self.model.base_model_prefix)
        self.head = self.model.get_output_embeddings()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        self.budget = token_budget
        # Each letter is scored as "A" or " A" (log-sum-exp of the two tokens).
        self.letter_ids = []
        for variant in ("{}",) if protocol == "noprefill" else ("{}", " {}"):
            ids = []
            for letter in LETTERS[:5]:
                enc = self.tok.encode(variant.format(letter), add_special_tokens=False)
                assert len(enc) == 1, f"{variant.format(letter)!r} is not a single token: {enc}"
                ids.append(enc[0])
            self.letter_ids.append(ids)
        # Some models open the answer with a markdown bold marker (" **A**"); see run_letters.
        enc = self.tok.encode(BOLD, add_special_tokens=False)
        assert len(enc) == 1, f"{BOLD!r} is not a single token: {enc}"
        self.bold_id = enc[0]

    def chat_text(self, prompt, lang):
        """Chat prompt with the assistant turn opened by an answer prefix ("정답:" / "Answer:")."""
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if self.protocol == "noprefill":
            return text
        return text + PROMPTS[lang]["answer_prefix"]

    def chat_ids(self, prompt, lang, bold=False):
        text = self.chat_text(prompt, lang) + (BOLD if bold else "")
        return self.tok.encode(text, add_special_tokens=False)[-MAX_TOKENS:]

    def batches(self, seqs):
        """Yield index batches of similar length within the token budget."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        batch, longest = [], 0
        for i in order:
            longest_new = max(longest, len(seqs[i]))
            if batch and longest_new * (len(batch) + 1) > self.budget:
                yield batch
                batch, longest_new = [], len(seqs[i])
            batch.append(i)
            longest = longest_new
        if batch:
            yield batch

    @torch.no_grad()
    def hidden(self, seqs):
        """Right-padded forward pass; returns last hidden states and lengths."""
        lengths = [len(s) for s in seqs]
        width = max(lengths)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for row, s in enumerate(seqs):
            ids[row, : len(s)] = torch.tensor(s)
            mask[row, : len(s)] = 1
        out = self.base(input_ids=ids.cuda(), attention_mask=mask.cuda())
        return out[0], lengths

    @torch.no_grad()
    def letter_logprobs(self, seqs):
        """Next-token log-probabilities of the five letters ("A" or " A") and of the bold marker."""
        result = [None] * len(seqs)
        for batch in self.batches(seqs):
            h, lengths = self.hidden([seqs[i] for i in batch])
            last = torch.stack([h[row, n - 1] for row, n in enumerate(lengths)])
            logp = torch.log_softmax(self.head(last).float(), dim=-1)
            letters = torch.logsumexp(torch.stack([logp[:, ids] for ids in self.letter_ids]), dim=0)
            bold = logp[:, self.bold_id]
            for row, i in enumerate(batch):
                result[i] = letters[row].tolist() + [bold[row].item()]
        return result

    @torch.no_grad()
    def token_logprobs(self, seqs, starts):
        """Log-probabilities of tokens seqs[i][starts[i]:] given their prefix."""
        result = [None] * len(seqs)
        for batch in self.batches(seqs):
            h, lengths = self.hidden([seqs[i] for i in batch])
            for row, i in enumerate(batch):
                start, n = starts[i], lengths[row]
                target = torch.tensor(seqs[i][start:n], device=h.device)
                values = []
                for c in range(start, n, 512):
                    end = min(c + 512, n)
                    logp = torch.log_softmax(self.head(h[row, c - 1 : end - 1]).float(), dim=-1)
                    values.append(logp.gather(1, target[c - start : end - start, None]).squeeze(1))
                result[i] = torch.cat(values).tolist()
        return result


def run_letters(scorer, items, condition):
    """Letter probabilities marginalised over two answer formats after the answer prefix:
    P(L) = P(L | prefix) + P(" **" | prefix) * P(L | prefix + " **"), renormalised over the k letters.
    The earlier protocols (noprefill, prefix_only) use the first term only."""
    with_bold = scorer.protocol == "final"
    plain, bold, keys = [], [], []
    for i, item in enumerate(items):
        for r in range(len(item["options"])):
            prompt = build_prompt(item, condition, r)
            plain.append(scorer.chat_ids(prompt, item["lang"]))
            if with_bold:
                bold.append(scorer.chat_ids(prompt, item["lang"], bold=True))
            keys.append((i, r))
    first = scorer.letter_logprobs(plain)
    after_bold = scorer.letter_logprobs(bold) if with_bold else [None] * len(first)
    rows = [{"uid": item["uid"], "rotations": [None] * len(item["options"]), "mass": [None] * len(item["options"])}
            for item in items]
    for (i, r), a, b in zip(keys, first, after_bold):
        k = len(items[i]["options"])
        logp = torch.tensor(a[:k])
        if with_bold:
            logp = torch.logaddexp(logp, a[5] + torch.tensor(b[:k]))
        rows[i]["rotations"][r] = [round(p, 6) for p in torch.softmax(logp, dim=0).tolist()]
        rows[i]["mass"][r] = round(torch.logsumexp(logp, dim=0).exp().item(), 6)
    return rows


def run_option_alone(scorer, items):
    seqs, starts, keys = [], [], []
    for i, item in enumerate(items):
        prefix = scorer.bos + scorer.tok.encode(PROMPTS[item["lang"]]["option_prefix"], add_special_tokens=False)
        for j, option in enumerate(item["options"]):
            body = scorer.tok.encode(str(option), add_special_tokens=False) or [scorer.pad_id]
            seqs.append((prefix + body)[:MAX_TOKENS])
            starts.append(len(prefix))
            keys.append((i, j))
    logps = scorer.token_logprobs(seqs, starts)
    rows = [{"uid": item["uid"], "scores": [None] * len(item["options"])} for item in items]
    for (i, j), lp in zip(keys, logps):
        rows[i]["scores"][j] = round(sum(lp) / len(lp), 6)
    return rows


def run_mink(scorer, items):
    seqs = [(scorer.bos + scorer.tok.encode(item_text(item), add_special_tokens=False))[:MAX_TOKENS] for item in items]
    logps = scorer.token_logprobs(seqs, [1] * len(seqs))
    rows = []
    for item, lp in zip(items, logps):
        lowest = sorted(lp)[: max(1, int(len(lp) * MINK_RATIO))]
        rows.append({"uid": item["uid"], "mink": round(sum(lowest) / len(lowest), 6), "n_tokens": len(lp)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--items", default="data/items.jsonl")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tasks", default="full,choices_only,option_alone,mink")
    parser.add_argument("--limit", type=int, default=0, help="random subset for a pilot run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="final")
    args = parser.parse_args()

    items = read_jsonl(args.items)
    out_root = args.out if args.protocol == "final" else os.path.join(args.out, args.protocol)
    out_dir = os.path.join(out_root, args.model)
    if args.limit:
        items = random.Random(args.seed).sample(items, args.limit)
        out_dir = os.path.join(out_root, f"pilot_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    scorer = Scorer(args.model, args.token_budget, args.protocol)
    for task in args.tasks.split(","):
        path = os.path.join(out_dir, f"{task}.jsonl")
        if os.path.exists(path):
            print(f"{task}: exists, skipped")
            continue
        start = time.time()
        if task in ("full", "choices_only"):
            rows = run_letters(scorer, items, task)
        elif task == "option_alone":
            rows = run_option_alone(scorer, items)
        elif task == "mink":
            rows = run_mink(scorer, items)
        else:
            raise SystemExit(f"unknown task {task}")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(path + ".tmp", path)
        print(f"{task}: {len(rows)} items in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
