"""GGUF 계약 fixture 생성 — C 엔진이 Python 없이 같은 계산을 하는지 확인하는 기대값

    python -m src.export_fixtures            # cli/fixtures/v<CONTRACT_VERSION>/ 의 cases.txt, manifest.json 갱신
    python -m src.export_fixtures --new-model  # 작은 무작위 모델(model.gguf)부터 새로 만든다 (계약 버전을 올렸을 때)

기대값은 항상 커밋된 model.gguf 를 읽어 Python 으로 계산한다. 그래서 무작위 초기화가 torch 버전마다 달라도
fixture 는 바뀌지 않고, 계산 규칙(cli/CONTRACT.md)이 바뀌었을 때만 바뀐다.
tests/test_contract_fixtures.py 가 이 값을 다시 계산해 커밋된 값과 비교한다.

cases.txt 형식 (탭 구분, 문자열은 UTF-8 바이트의 hex, 빈 문자열은 빈 칸):
    encode    <text>    <token id ...>
    logits    <token id ...>    <logit ...>
    generate  <max_new_tokens>    <question>    <output>    <stop: eos | max_new_tokens | context>
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.exporter import export_model, read_gguf
from src.inference.gguf_model import InferenceModel, ModelConfig
from src.tokenizer import bpe
from src.tokenizer.contract import CONTRACT_VERSION

FIXTURE_ROOT = Path("cli/fixtures")
MODEL_FILE = "model.gguf"
CASES_FILE = "cases.txt"
MANIFEST_FILE = "manifest.json"

BOS_ID, EOS_ID, SEP_ID = (
    bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index(token)
    for token in ("<bos>", "<eos>", "<sep>")
)

# context 를 작게 두어 문맥 길이 초과 경로도 짧은 질문으로 확인한다.
FIXTURE_CONFIG = ModelConfig(
    vocab_size=320, dim=16, heads=2, layers=2, ffn_dim=24, context=32
)
FIXTURE_CORPUS = ["hello hello", "stock mouse", "SELECT stock FROM items;"]

ENCODE_TEXTS = [
    "hello hello",
    "SELECT stock FROM items;",
    "new york_ new york",
    "customers.name = 'mouse'",
    "123 45.6\n",
    "stockholm stock",
    "",
    " ",
    "\t\n",
    "stock stock",
    "stock_ stockholm",
    "1stock",
    "_stock",
    "aaaaaa",
]
LOGITS_IDS = [
    [BOS_ID],
    [BOS_ID, 104, 105, SEP_ID],
    [BOS_ID, 1, 2, 3, 4, 5, SEP_ID],
    [BOS_ID] + [104] * (FIXTURE_CONFIG.context - 1),  # 문맥을 꽉 채운 입력
]
GENERATE_CASES = [
    ("hello", 5),
    ("hello stock", 3),
    ("HELLO STOCK", 3),  # 질문은 소문자로 바꿔 인코딩한다
    ("", 0),
    ("hello", 32),
    ("1" * (FIXTURE_CONFIG.context - 2), 8),  # <bos> 질문 <sep> 이 문맥을 꽉 채움
]


def new_fixture_model(path: Path) -> None:
    torch.manual_seed(17)
    model = InferenceModel(FIXTURE_CONFIG).eval()
    seeds = bpe.load_seed_tokens()
    tokenizer = bpe.Tokenizer.train(
        FIXTURE_CORPUS, seeds, vocab_size=bpe.SEED_BASE + len(seeds) + 12
    )
    export_model(model, path, tokenizer.merges)


def last_logits(model, ids: list[int]) -> list[float]:
    with torch.no_grad():
        return model(torch.tensor([ids]))[0, -1].tolist()


def reference_generate(model, tokenizer, question: str, max_new_tokens: int):
    """C 엔진과 같은 규칙의 greedy 생성: 바이트가 있는 토큰만 후보, 동점이면 <eos> 다음 작은 ID."""
    ids = [BOS_ID] + tokenizer.encode(question.lower()) + [SEP_ID]
    candidates = sorted(tokenizer.id_to_bytes)
    output = bytearray()
    for _ in range(max_new_tokens):
        if len(ids) >= model.config.context:
            return bytes(output), "context"
        logits = last_logits(model, ids)
        next_id = EOS_ID
        for candidate in candidates:
            if logits[candidate] > logits[next_id]:
                next_id = candidate
        if next_id == EOS_ID:
            return bytes(output), "eos"
        output.extend(tokenizer.id_to_bytes[next_id])
        ids.append(next_id)
    return bytes(output), "max_new_tokens"


def compute_cases(model_path: Path) -> list[tuple]:
    model, tokenizer, _ = read_gguf(model_path)
    cases = [("encode", text, tokenizer.encode(text)) for text in ENCODE_TEXTS]
    cases += [("logits", ids, last_logits(model, ids)) for ids in LOGITS_IDS]
    for question, max_new_tokens in GENERATE_CASES:
        output, stop = reference_generate(model, tokenizer, question, max_new_tokens)
        cases.append(("generate", max_new_tokens, question, output, stop))
    return cases


def _hex(value) -> str:
    return (value.encode("utf-8") if isinstance(value, str) else value).hex()


def _ints(values) -> str:
    return " ".join(map(str, values))


def format_cases(cases: list[tuple]) -> str:
    lines = [
        f"# capstone_sql contract v{CONTRACT_VERSION} fixtures; see src/export_fixtures.py"
    ]
    for case in cases:
        kind = case[0]
        if kind == "encode":
            fields = [_hex(case[1]), _ints(case[2])]
        elif kind == "logits":
            fields = [_ints(case[1]), " ".join(f"{value:.9g}" for value in case[2])]
        else:
            fields = [str(case[1]), _hex(case[2]), _hex(case[3]), case[4]]
        lines.append("\t".join([kind, *fields]))
    return "\n".join(lines) + "\n"


def parse_cases(text: str) -> list[tuple]:
    cases = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        kind, *fields = line.split("\t")
        if kind == "encode":
            ids = [int(value) for value in fields[1].split()]
            cases.append((kind, bytes.fromhex(fields[0]).decode("utf-8"), ids))
        elif kind == "logits":
            ids = [int(value) for value in fields[0].split()]
            cases.append((kind, ids, [float(value) for value in fields[1].split()]))
        else:
            question = bytes.fromhex(fields[1]).decode("utf-8")
            cases.append(
                (kind, int(fields[0]), question, bytes.fromhex(fields[2]), fields[3])
            )
    return cases


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GGUF 계약 fixture 생성")
    ap.add_argument(
        "--output",
        default=str(FIXTURE_ROOT / f"v{CONTRACT_VERSION}"),
        help=f"fixture 폴더 (기본 {FIXTURE_ROOT}/v{CONTRACT_VERSION})",
    )
    ap.add_argument(
        "--new-model", action="store_true", help="model.gguf 를 새로 만든다"
    )
    args = ap.parse_args(argv)

    output = Path(args.output)
    model_path = output / MODEL_FILE
    if args.new_model or not model_path.exists():
        new_fixture_model(model_path)
    cases_path = output / CASES_FILE
    cases_path.write_bytes(format_cases(compute_cases(model_path)).encode("utf-8"))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "files": {name: sha256(output / name) for name in (MODEL_FILE, CASES_FILE)},
    }
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {output} (contract v{CONTRACT_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
