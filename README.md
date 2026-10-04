# 🤖 Text-to-SQL Transformer LLM (Scratch Build)

4인 캡스톤 디자인 프로젝트: 밑바닥(Scratch)부터 구현하는 경량 Text-to-SQL Transformer LLM 저장소입니다.

자연어 질문(예: `how many mouse units are left in stock ?`)을 입력받아 정확한 SQLite SQL 쿼리(`SELECT stock FROM items WHERE item_name = 'mouse';`)로 변환하는 언어 모델을 구현합니다.

---

## 📁 프로젝트 구조 (src 표준 구조)

```text
capstone-design-1/
├── src/                                  # 핵심 소스코드 패키지
│   ├── __init__.py
│   ├── config.py                         # 모델/학습 하이퍼파라미터
│   ├── train.py                          # 학습 루프 (runs/ 에 체크포인트·로그 저장)
│   ├── evaluation.py                     # 체크포인트 평가 (EM, 실행 정확도, 계층별 집계)
│   ├── predict.py                        # 질문 → SQL 생성·실행 (대화형)
│   ├── models/                           # 트랜스포머 모델 아키텍처
│   │   ├── __init__.py
│   │   ├── embedding.py                  # TokenEmbedding, RoPE
│   │   ├── transformer.py                # RMSNorm, Attention, SwiGLU FFN, 디코더 블록
│   │   └── model.py                      # 전체 모델 조립, greedy 생성
│   ├── tokenizer/                        # 토크나이저 모듈
│   │   ├── __init__.py
│   │   └── bpe.py                        # 바이트 레벨 BPE 토크나이저
│   └── data/                             # 데이터셋 및 데이터 생성 모듈
│       ├── __init__.py
│       ├── dataset.py                    # PyTorch Dataset & collate_fn
│       ├── name_swap.py                  # 이름 교체 (가짜 이름 생성, 기본 학습에 사용)
│       ├── clean_ambiguous.py            # 질문만으로 정답이 정해지지 않는 학습 쌍 제거
│       └── generator/                    # SQL/질문 데이터 파이프라인
│           ├── __init__.py
│           ├── db_gen.py                 # SQLite DB 및 데이터 생성
│           ├── sql_gen.py                # SQL 쿼리 생성
│           ├── question_gen.py           # LLM 기반 자연어 질문 생성 v1 (수동 붙여넣기 방식)
│           └── question_gen_auto.py      # v2 — 로컬 Ollama 모델 호출 완전 자동화
├── data/                                 # 실제 데이터 산출물 (DB, JSON)
│   ├── shop.db                           # 평가/학습용 SQLite DB
│   ├── holdout.json                      # 미등장 평가용 엔티티
│   ├── sql_train.json                    # 학습용 SQL 목록
│   ├── sql_eval_indist.json              # 분포 내 평가용 SQL 목록
│   ├── sql_eval_holdout.json             # 미등장 값 평가용 SQL 목록
│   ├── sql_gen_report.json               # SQL 구조별 할당량 보고서
│   ├── pilot_merged/                     # 학습용 (질문, SQL) 페어 데이터 (source 필드로 출처 구분)
│   ├── eval_indist_merged/               # 분포 내 평가용 (질문, SQL) 페어 데이터
│   ├── eval_holdout_merged/              # 미등장 값 평가용 (질문, SQL) 페어 데이터
│   └── eval_qwen_holdout_merged/         # 이전 미등장 값 평가셋 (Qwen3 생성, 비교용)
├── tests/                                # 전체 단위 테스트 (pytest)
│   ├── conftest.py                       # pytest 환경 설정 (sys.path)
│   ├── test_embedding.py                 # 임베딩 및 RoPE 검증
│   ├── test_transformer.py               # 디코더 블록 검증
│   ├── test_tokenizer.py                 # BPE 토크나이저 검증
│   ├── test_dataset.py                   # PyTorch Dataset 검증
│   ├── test_model.py                     # 모델 조립·causal·weight tying·greedy 생성 검증
│   ├── test_train.py                     # SQL 단위 분할·loss 마스크·lr 스케줄 검증
│   ├── test_evaluation.py                # 채점 규칙(EM·다중 정답·오답 분류)·배치 생성 검증
│   ├── test_predict.py                   # 질문 → SQL 생성·실행 출력 검증
│   ├── test_name_swap.py                 # 가짜 이름 생성·이름 교체 검증
│   ├── test_clean_ambiguous.py           # 라벨 규칙(L2~L4) 위반 판정 검증
│   ├── test_question_gen.py              # 질문 생성 검증 로직 검증
│   └── test_question_gen_auto.py         # 자동화 파이프라인 JSON 복구 로직 검증
├── docs/                                 # 프로젝트 설계 및 규칙 문서
│   ├── CONVENTION.md                     # 팀 협업 규칙 및 커밋 컨벤션
│   ├── SETUP.md                          # 환경 설정 가이드
│   ├── text-to-sql-llm-project-plan.md   # 전체 프로젝트 상세 설계서
│   └── text-to-sql-stage1-report.md      # 1단계 진행 보고서
├── CLAUDE.md                             # Claude Code 작업 규칙
├── pytest.ini                            # pytest 마커·옵션
├── requirements.txt                      # 프로젝트 통합 의존성
└── README.md
```

---

## ⚙️ 환경 설정 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```
> **참고**: PyTorch CUDA 빌드 인덱스(`--extra-index-url`)가 `requirements.txt`에 포함되어 있습니다.

### 2. 전체 단위 테스트 실행
프로젝트 루트에서 다음 명령으로 모든 모듈의 테스트를 한 번에 검증할 수 있습니다:
```bash
pytest tests/ -v
```


---

## 📜 협업 규칙 및 문서
- 팀 협업 및 Git 커밋/PR 규칙은 [docs/CONVENTION.md](docs/CONVENTION.md)를 확인하세요.
- 전체 데이터셋 구조 및 모델 아키텍처 계획은 [docs/text-to-sql-llm-project-plan.md](docs/text-to-sql-llm-project-plan.md)를 참고하세요.
- 현재 진행 상황과 결과는 [docs/text-to-sql-stage1-report.md](docs/text-to-sql-stage1-report.md), 실행·재현 방법은 [docs/SETUP.md](docs/SETUP.md)에 있습니다.