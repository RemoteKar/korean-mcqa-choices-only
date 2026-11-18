"""Shared prompt templates and helpers."""
import json

LETTERS = "ABCDEFGHIJ"

PROMPTS = {
    "ko": {
        "full": "다음 객관식 문제를 읽고 정답을 고르세요. 정답 선택지의 기호({letters})만 답하세요.\n\n{context}질문: {question}\n{options}",
        "choices_only": "다음은 어떤 객관식 문제의 선택지입니다. 질문은 주어지지 않았습니다. 정답일 가능성이 가장 높은 선택지를 고르세요. 선택지의 기호({letters})만 답하세요.\n\n{options}",
        "context": "지문: {paragraph}\n\n",
        "option_prefix": "정답: ",
        "answer_prefix": "정답:",
    },
    "en": {
        "full": "Read the following multiple-choice question and choose the correct answer. Answer with the letter of the correct option ({letters}) only.\n\n{context}Question: {question}\n{options}",
        "choices_only": "The following are the options of a multiple-choice question. The question is not shown. Choose the option most likely to be correct. Answer with the letter of the option ({letters}) only.\n\n{options}",
        "context": "Passage: {paragraph}\n\n",
        "option_prefix": "Answer: ",
        "answer_prefix": "Answer:",
    },
}


def rotate(options, r):
    """Cyclic rotation: position j shows original option (j + r) mod k."""
    k = len(options)
    return [options[(j + r) % k] for j in range(k)]


def original_index(position, r, k):
    return (position + r) % k


def format_options(options):
    return "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))


def build_prompt(item, condition, r):
    p = PROMPTS[item["lang"]]
    opts = rotate(item["options"], r)
    letters = ", ".join(LETTERS[: len(opts)])
    context = p["context"].format(paragraph=item["paragraph"]) if item.get("paragraph") else ""
    return p[condition].format(letters=letters, context=context, question=item["question"], options=format_options(opts))


def item_text(item):
    """Raw text of the full item (no instruction, no answer), used for Min-K% Prob."""
    parts = []
    if item.get("paragraph"):
        parts.append(item["paragraph"])
    parts.append(item["question"])
    parts.append(format_options(item["options"]))
    return "\n".join(parts)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
