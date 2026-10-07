# 🤖 Text-to-SQL Transformer LLM (Scratch Build)

4인 캡스톤 디자인 프로젝트: 밑바닥(Scratch)부터 구현하는 경량 Text-to-SQL Transformer LLM 저장소입니다.

자연어 질문(예: `how many mouse units are left in stock?`)을 입력받아 SQLite SQL 쿼리(`SELECT stock FROM items WHERE item_name = 'mouse'`)로 변환하는 언어 모델을 구현합니다. 모델, 토크나이저, 학습 루프, 추론을 모두 직접 구현하고 사전학습 가중치는 쓰지 않습니다.

---

## 🚀 바로 써 보기

학습된 배포 모델(`models/stage1/`)이 저장소에 포함되어 있어 학습 없이 바로 실행할 수 있습니다.

```bash
pip install -r requirements.txt
python -m src.run --input "what city does ashley live in?"   # 질문 → SQL → DB 실행 결과
python -m src.run                                            # 대화형
python -m src.run --evaluation                               # 평가셋 전부 채점
python -m src.demo all                                       # 예시 하나로 파이프라인 각 단계의 입력 → 출력 보기
pytest tests/ -v                                             # 단위 테스트
```

> **참고**: PyTorch CUDA 빌드 인덱스(`--extra-index-url`)가 `requirements.txt`에 포함되어 있습니다. 추론·평가는 CPU로도 됩니다(`--cpu`).
> 모든 명령은 저장소 루트에서 실행합니다. 학습·데이터 생성·인자 설명은 [docs/SETUP.md](docs/SETUP.md)에 있습니다.

---

## 📁 프로젝트 구조

```text
capstone-design-1/
├── src/
│   ├── config.py                         # 단계별 모델 구조, 학습 설정, 데이터·모델 경로 (STAGE 하나로 전환)
│   ├── train.py                          # 학습 루프 → runs/stage<N>/<run>/
│   ├── release.py                        # 학습 체크포인트 → 배포 모델 폴더 models/stage<N>/
│   ├── run.py                            # 추론·평가 통합 명령 (단건 / JSON 배치 / 대화형, --test / --evaluation)
│   ├── demo.py                           # 파이프라인 단계별 입력 → 출력 보기
│   ├── models/
│   │   ├── embedding.py                  # TokenEmbedding, RoPE
│   │   ├── transformer.py                # RMSNorm, Attention, SwiGLU FFN, 디코더 블록
│   │   └── model.py                      # 전체 모델 조립, greedy 생성
│   ├── tokenizer/
│   │   └── bpe.py                        # 바이트 레벨 BPE 토크나이저 (Tokenizer)
│   ├── inference/                        # 학습된 모델 쓰기
│   │   ├── loading.py                    # 체크포인트·배포 모델 불러오기, 짝이 맞는 토크나이저 찾기
│   │   └── generate.py                   # 질문 → SQL greedy 생성 (배치), 디코딩
│   ├── evaluation/                       # 평가
│   │   ├── scoring.py                    # 채점·집계: EM, 다중 정답 EM, 규칙 일치 EM, 실행 정확도, 신뢰구간
│   │   └── eval_sets.py                  # 평가셋 폴더(pairs.json + sql.json), 평가 결과 위치
│   └── data/
│       ├── dataset.py                    # <bos> 질문 <sep> SQL <eos> 토큰화, loss_mask, collate_fn
│       ├── db_gen.py                     # SQLite DB·미등장 값 생성 (스키마가 같은 단계 공통)
│       └── stage1/                       # 1단계 전용
│           ├── sql_gen.py                # SQL 생성 (학습 / 분포 내 / 미등장 값)
│           ├── question_gen.py           # LLM 질문 생성 v1 (수동 전달) · 검증
│           ├── question_gen_auto.py      # v2 — 로컬 Ollama 모델 호출 자동화
│           ├── clean_ambiguous.py        # 라벨 규칙(L2~L4) 위반 학습 쌍 제거
│           ├── name_swap.py              # 이름 교체 (가짜 이름 생성, 기본 학습에 사용)
│           └── sql_rules.py              # 1단계 채점 규칙 (SQL 분해, 오답 분류, 다중 정답, 규칙 일치)
├── data/
│   ├── stage1/
│   │   ├── db/                           # shop.db, holdout.json (미등장 값)
│   │   ├── sql/                          # train.json, eval_indist.json, eval_holdout.json, gen_report.json
│   │   ├── train/                        # 학습 (질문, SQL) 쌍 — 이 폴더의 *.json 전부가 학습 데이터·BPE 코퍼스
│   │   ├── eval/                         # 평가셋마다 폴더 하나: indist/, holdout/, holdout_qwen/ (pairs.json + sql.json)
│   │   └── tokenizer/                    # seed_tokens.json (seed 토큰), tokenizer.json (BPE 학습 결과)
│   └── wordnet/                          # 이름 교체용 영단어 목록 (WordNet 3.0, 라이선스 포함)
├── models/
│   └── stage1/                           # 배포 모델: model.pt, tokenizer.json, model_info.json
├── tests/                                # pytest — src/ 와 같은 구조 (tests/inference/, tests/evaluation/, tests/data/stage1/, ...)
├── docs/
│   ├── CONVENTION.md                     # 팀 협업 규칙 및 커밋 컨벤션
│   ├── SETUP.md                          # 환경 설정, 각 단계 실행·재현 방법, 인자 설명
│   ├── text-to-sql-llm-project-plan.md   # 전체 프로젝트 설계서
│   └── text-to-sql-stage1-report.md      # 1단계 진행 보고서
├── CLAUDE.md                             # Claude Code 작업 규칙
├── pytest.ini                            # pytest 마커·옵션
├── requirements.txt                      # 프로젝트 통합 의존성
└── README.md
```

커밋하지 않는 폴더: `data_raw/`(질문 생성 라운드·중간 산출물), `runs/`(학습·평가·추론 결과).

---

## 📜 협업 규칙 및 문서
- 팀 협업 및 Git 커밋/PR 규칙은 [docs/CONVENTION.md](docs/CONVENTION.md)를 확인하세요.
- 데이터셋 구조와 모델 아키텍처 설계는 [docs/text-to-sql-llm-project-plan.md](docs/text-to-sql-llm-project-plan.md)를 참고하세요.
- 현재 진행 상황과 결과는 [docs/text-to-sql-stage1-report.md](docs/text-to-sql-stage1-report.md), 실행·재현 방법은 [docs/SETUP.md](docs/SETUP.md)에 있습니다.
