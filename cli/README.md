# capstone_sql C 추론 엔진 · SQL 후처리기

영어 질문 → SQLite SQL을 생성하는 스크래치 Transformer의 C 실행부다. 외부 런타임 라이브러리 없이 학습 저장소가 내보낸
GGUF 파일 하나로 동작한다. 모델 구조와 토크나이저는 파일에서 읽고, 파일에 담을 수 없는 계산 규칙은
[CONTRACT.md](CONTRACT.md)에 있다.

| 경로 | 내용 |
| --- | --- |
| `engine/loader.c` | GGUF 로더 + forward + BPE + greedy 생성. 한 파일로 빌드된다 |
| `postprocessor/` | 생성된 SQL의 예약어·식별자 오타 교정 라이브러리와 CLI ([README](postprocessor/README.md)) |
| `fixtures/v<N>/` | 계약 버전별 작은 테스트 모델과 기대값 (CONTRACT.md 7절) |
| `tests/fixture_test.c` | fixture 기대값과 엔진 계산을 비교하는 테스트 |

## 빌드와 테스트

GCC(또는 C11 컴파일러)와 make가 필요하다. 이 폴더에서 실행한다(저장소 루트에서는 `make -C cli ...`).

```bash
make            # build/loader, postprocessor/ 의 라이브러리와 CLI
make test       # fixture 테스트(지원하는 계약 버전 전부) + 후처리기 단위 테스트
make clean
```

make 없이 엔진만 빌드할 때는 한 줄이면 된다.

```bash
gcc -std=c11 -O2 -Wall -Wextra -Werror engine/loader.c -lm -o loader
```

## 실행

```bash
./build/loader MODEL.gguf --generate "what city does ashley live in?" 64   # 질문 → SQL (stderr 에 종료 이유)
./build/loader MODEL.gguf --encode "SELECT stock FROM items;"              # 토큰 ID
./build/loader MODEL.gguf --logits 257 104 105 259                        # 마지막 위치 logits
```

Windows에서는 `build\loader.exe`다. 배포 모델은 학습 저장소의 GitHub Release(태그 `stage<N>-contract<V>`)에
`model.gguf`, `shop.db`, `SHA256SUMS`로 올라온다. 이 엔진이 지원하는 계약 버전의 태그만 쓴다.

## 코드 구조

처음 읽을 때는 `main` → `dispatch_cli_command` → 모드별 실행 함수를 따라가고, 생성 모드에서는
`prepare_prompt_tokens` → `generate_tokens` → `forward` 순서로 본다. 파일 로드는 `load_model`에서 단계별 함수를
따라간다.

| 역할 | 주요 함수 |
| --- | --- |
| 프로그램 시작 및 정리 | `main`, `release_model`, `release_tokenizer` |
| CLI 옵션 해석 및 분기 | `parse_cli_options`, `dispatch_cli_command` |
| 모드별 실행 | `run_logits_command`, `run_encode_command`, `run_generation_command` |
| 모델 파일 읽기 | `read_model_file`, `read_gguf_header`, `read_model_metadata` |
| 텐서 로드 및 검증 | `read_tensor_descriptors`, `read_tensor_values`, `validate_model_weights` |
| 토큰 바이트 복원 | `build_token_pieces`, `build_merged_token_piece` |
| 질문 준비 및 토큰화 | `normalize_question`, `prepare_prompt_tokens`, `encode_text`, `encode_piece` |
| 추론 단계 실행 | `forward`, `run_decoder_block`, `compute_last_token_logits` |
| Attention 및 FFN | `compute_queries_keys_values`, `compute_causal_attention`, `apply_attention_residual`, `apply_ffn_residual` |
| 다음 토큰 생성 | `generate_tokens`(ID 추가), `select_next_token` |
| 추론 임시 메모리 | `create_workspace`, `release_workspace` |

`ModelConfig`는 구조 설정, `Tokenizer`는 BPE와 토큰 바이트, `InferenceWorkspace`는 계산용 임시 버퍼를 묶는다.
복잡한 조건은 `is_model_config_compatible`, `is_tensor_data_in_bounds`, `has_valid_merge_references` 등으로
분리했고, 단락 평가로 잘못된 배열 접근과 0으로 나누기를 막는 순서를 유지한다.

## 한계

- 배치 1이고 매 생성 단계 전체 prefix를 다시 계산한다. KV cache, 샘플링, 양자화, Unicode 입력, SQL 문법 제약은 없다.
- 파일 및 개별 할당은 1 GiB 제한, layers ≤ 256, context ≤ 4096. 할당 합계에 대한 전역 메모리 예산은 없다.

## 코드 작성 기준

포맷은 [.clang-format](.clang-format)(들여쓰기 공백 4칸)을 따른다.

- `for`, `while`, `if`, `else`는 본문이 한 줄이어도 중괄호를 쓴다. 한 줄에 여러 실행문을 넣지 않는다.
- 함수와 변수는 `snake_case`, 타입은 `ModelConfig`처럼 역할이 드러나는 이름, 상수는 `GGUF_ALIGNMENT`처럼
  `UPPER_SNAKE_CASE`.
- `nt`, `nm`, `d` 대신 `tensor_count`, `merge_count`, `embedding_dim`처럼 개수와 의미를 구분할 수 있는 이름을 쓴다.
- 입력 파싱, 파일 읽기, 모델 계산, 출력 처리는 서로 다른 함수로 나눈다. `main()`과 상위 함수는 단계별 함수를
  호출해 흐름을 보여준다.
- 읽기 전용 모델과 가중치는 `const`로 표시하고, 할당한 메모리의 해제 책임은 같은 역할의 생성/해제 함수로 짝짓는다.
- 주석은 단락 평가 순서, 버퍼 범위, 수식 대응처럼 코드만으로 드러나지 않는 이유와 제약을 설명한다.

```bash
clang-format -i engine/loader.c                     # 포맷 적용
clang-format --dry-run --Werror engine/loader.c     # 검사만
```

## 검증 기록

- 2026-10-03: 더미 가중치로 Python/C logits(`atol = rtol = 2e-5`), 생성 반복, EOS·문맥 종료, BPE, 잘못된 입력·파일
  검증. 선택적 `gguf.GGUFReader`로 텐서 값을 독립 확인했다.
- 2026-10-06: 가독성 리팩터링(위 코드 구조). C 실행이 필요한 테스트 53개는 Windows 애플리케이션 제어
  정책(`WinError 4551`)으로 실행이 막혀 그날은 확인하지 못했다(10-07 재검증).
- 2026-10-07: 계약 v2. seed 개수 0/2/45, 바뀐 특수 토큰 ID와 seed/merge 시작 ID, 잘못된 ID 범위·충돌·병합 참조,
  계약 v1 거부를 검증했다. `stock_` 토큰화 불일치를 찾아 고쳤다(CONTRACT.md 4절 3번).
- 2026-10-12: `cli/`로 분리. Python 없이 도는 fixture 테스트(`make test`) 추가. `generate_tokens`는 생성한 ID를
  추가만 하고 출력은 `run_generation_command`가 한다(CLI 출력은 같다).
