# GGUF Exporter / C inference engine

## 구현 범위

기존 #9의 EXL1 확인용 코드를 GGUF v3와 실제 C forward로 교체했다.
무작위 가중치 기준으로 Python/C logits와 반복 생성 결과를 비교한다.
학습된 체크포인트가 아직 없어 SQL 정확도는 검증하지 않았다.

`src/inference.py`는 팀의 `TokenEmbedding`, `TransformerDecoderBlock`,
`RMSNorm`을 사용하여 전체 모델을 감싼 검증용 모델이다.
전체 학습 모델이 완성되면 이 문서의 체크포인트 계약으로 연결하거나
Exporter의 이름 대응을 수정해야 한다. 파라미터 수만 같은 모델은 호환되지 않는다.

## 실행 (저장소 루트, Windows PowerShell)

PyTorch, NumPy, pytest 및 GCC가 필요하다. 독립 GGUF 검증에는
`pip install -r requirements-inference.txt`도 실행한다.

```powershell
python -m src.exporter --dummy --output dummy.gguf
gcc -std=c11 -O2 -Wall -Wextra -Werror src/loader.c -lm -o loader.exe
.\loader.exe dummy.gguf --encode "SELECT stock FROM items;"
.\loader.exe dummy.gguf --logits 257 104 105 259
.\loader.exe dummy.gguf --generate "how many mouse units are left in stock?" 20
python -m pytest tests/test_inference.py -q
```

Linux에서는 `-o loader`로 빌드하고 `./loader`로 실행한다.
`--dummy`는 dim=16, heads=2, layers=2, ffn_dim=32의 무작위 모델이다.
더미 기본 BPE는 병합 없이 바이트와 seed만 사용한다.
학습된 BPE를 사용하려면 `--merges bpe_merges.json`을 지정한다.
더미 출력은 SQL이거나 유효한 UTF-8 문자열이라는 보장이 없다.

## 실제 체크포인트 계약

```python
from dataclasses import asdict
import torch
from src.inference import ModelConfig, InferenceModel

config = ModelConfig()  # 최종 학습 설정을 정확히 전달
model = InferenceModel(config)
# 학습 루프에서 이 모델을 학습한 뒤 저장
torch.save({"config": asdict(config), "state_dict": model.state_dict()}, "model.pt")
```

```powershell
python -m src.exporter --checkpoint model.pt --merges bpe_merges.json --output model.gguf
.\loader.exe model.gguf --generate "how many mouse units are left in stock?" 64
```

최종 학습 모델이 다른 이름이나 `model_state_dict` 키를 쓰면 자동으로
추측하지 않고 어댑터를 추가한다. 현재 코드는 키와 shape를 엄격히 검사한다.
가중치가 공유된 모델은 현재 독립 embedding/lm_head 텐서 두 개로 내보낸다.

## 파일 형식과 이름 대응

[GGUF 공식 명세](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)를 따른
little-endian v3 컨테이너, F32(`GGML_TYPE_F32=0`), alignment=32를 사용한다.
Python writer에는 GGUF 라이브러리가 필요하지 않고 C에는 외부 라이브러리가 없다.

| 설정 | 메타데이터 |
| --- | --- |
| 구조 | `general.architecture = capstone_sql` |
| 계약 버전 | `capstone_sql.contract_version = 1` |
| 단어장/차원/층/헤드/FFN/문맥 | `capstone_sql.vocab_size/dim/layers/heads/ffn_dim/context` |
| RMSNorm/RoPE | `capstone_sql.eps/theta` |
| seed 토큰 | `capstone_sql.tokenizer.seeds`: string array |
| 순서 있는 BPE 병합 | `capstone_sql.tokenizer.merges`: uint32 array `[a,b,new_id,...]` |

계약 버전 1은 이 13개 메타데이터 항목을 사용한다. 임의의 GGUF를 받는
범용 엔진이 아니며 다른 구조, F16/양자화, 다른 메타데이터 계약은 거절한다.
`capstone_sql`은 사용자 정의 구조이므로 llama.cpp에서 자동 실행되지 않는다.

텐서 이름은 Python state_dict 그대로다. 텐서 데이터는 PyTorch의
`[out,in]` row-major이며 GGUF 차원 설명만 fastest-first 순서로 뒤집는다.

| 텐서 | Python shape |
| --- | --- |
| `embedding.embedding.weight` | `[vocab_size, dim]` |
| `blocks.N.attn_norm.weight`, `blocks.N.ffn_norm.weight` | `[dim]` |
| `blocks.N.attn.qkv_proj.weight` | `[3*dim, dim]`, Q→K→V 순서 |
| `blocks.N.attn.out_proj.weight` | `[dim, dim]` |
| `blocks.N.ffn.w1.weight`, `blocks.N.ffn.w3.weight` | `[ffn_dim, dim]` |
| `blocks.N.ffn.w2.weight` | `[dim, ffn_dim]` |
| `final_norm.weight` | `[dim]` |
| `lm_head.weight` | `[vocab_size, dim]` |

## C 계산 흐름

```text
GGUF 설정/전체 가중치 로드
→ 소문자 질문을 팀 BPE로 encode
→ [BOS=257] 질문 [SEP=259]
→ Embedding
→ 각 블록: RMSNorm → packed QKV → 인접 차원 RoPE
  → causal scaled attention → 출력 projection → residual
  → RMSNorm → SwiGLU → residual
→ final RMSNorm → lm_head → 마지막 위치 logits
→ greedy 선택 → EOS=258 또는 최대 길이까지 반복 → 토큰 바이트 출력
```

PAD=256, BOS=257, SEP=259, 병합 미정의 ID는 생성 후보에서 제외하고
EOS와 실제 복원 가능한 토큰만 고른다. 동점은 EOS 우선, 그 외 낮은 ID 우선이다.
입력은 1단계 영어 범위에 맞춰 ASCII만 지원하며 비ASCII는 명시적으로 거절한다.
숫자는 한 글자씩, seed는 단어 경계를 확인하고, 알파벳 조각에는 순서대로
BPE 병합을 적용한다. 이는 팀 regex tokenizer의 ASCII 동작과 비교 검증한다.

Baseline은 배치 1이고 매 생성 단계 전체 prefix를 다시 계산한다.
KV cache, 샘플링, 양자화, Unicode 입력, SQL 문법 제약은 후속 작업이다.
파일 및 개별 할당은 1 GiB 제한, layers<=256, context<=4096이며
각 할당 합계에 대한 전역 메모리 예산 관리 기능은 아직 없다.

## 검증과 다음 연동

`tests/test_inference.py`는 GCC 경고를 오류로 처리하여 빌드한다.
동일한 더미 가중치로 전체 Python/C logits를 `atol=rtol=2e-5` 기준 비교하고,
다른 설정, 토큰 생성 반복, EOS/문맥 종료, BPE 및 잘못된 입력/파일도 검증한다.
선택적 `gguf.GGUFReader` 검사로 모든 텐서 값의 파일 호환성을 독립 확인한다.

최종 학습 체크포인트가 제공되면 이름/설정/토크나이저 계약을 확인한 뒤
실제 질문의 Python/C logits와 SQL 결과를 추가 검증한다. 이는 현재 더미 검증과 구분한다.

2026-10-03 로컬 검증 결과: 새 추론 테스트 25개 포함 전체 99개 통과.
CPU PyTorch 2.14.1, GCC, GGUFReader 0.19.0으로 실행했다.
