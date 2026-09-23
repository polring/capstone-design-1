# datasets/

4인 캡스톤 팀 저장소(`capstone-design-1/`) 안에서 팀원1이 담당하는 범위: 데이터 생성 파이프라인 → BPE 토크나이저 →
PyTorch Dataset → (예정) `train.py`. 다른 팀원 폴더와 겹치지 않게 전부 이 폴더 밑에 있다.

## 문서 (`docs/`)

| 문서 | 설명 |
|---|---|
| [text-to-sql-llm-project-plan.md](docs/text-to-sql-llm-project-plan.md) | 전체 프로젝트 설계 문서 — 목표, 단계별 계획, DB 스키마, 설계 결정과 근거 (한국어, 가장 상세) |
| [text-to-sql-stage1-progress-report.md](docs/text-to-sql-stage1-progress-report.md) | 1단계 진행 상황 보고서 — 완료/보류 항목, 수치 요약 (한국어) |
| [function-reference.md](docs/function-reference.md) | `db_gen.py`/`sql_gen.py`/`question_gen.py`/`bpe_tokenizer.py`/`dataset.py` 함수별 입력·처리·출력 정리 (한국어) |
| [SETUP.md](docs/SETUP.md) | 환경을 처음부터 재현하는 가이드 — venv/PyTorch 설치, 스크립트별 실행 위치·순서, 뭘 커밋하고 뭘 안 하는지 |
| [CLAUDE.md](docs/CLAUDE.md) | 프로젝트 개요, 스크래치 빌드 제약, 현재 진행 상태, 실행 명령, 아키텍처 설명. |

## 파일 구조

```
datasets/
├─ .venv/                        가상환경 (커밋 안 함)
├─ requirements.txt              PyTorch(cu130)/NumPy/TensorBoard/pytest 고정
├─ bpe_tokenizer.py               BPE 토크나이저 (스크래치 구현)
├─ bpe_merges.json                학습된 병합 726개 (커밋 안 함 — 코드+시드로 재현)
├─ dataset.py                     PyTorch Dataset/collate_fn
├─ tests/                         pytest 단위 테스트
│  ├─ test_bpe_tokenizer.py       bpe_tokenizer.py 테스트
│  └─ test_dataset.py             dataset.py 테스트
├─ db/
│  ├─ db_gen.py                   DB 생성기 → shop.db, holdout.json
│  ├─ sql_gen.py                  SQL 생성기 → sql_train.json, sql_eval_indist.json
│  ├─ question_gen.py             LLM 질문 생성 파이프라인 (make-prompts / validate / template-check)
│  ├─ data/                       커밋되는 산출물
│  │  ├─ shop.db, holdout.json, sql_gen_report.json
│  │  ├─ sql_train.json, sql_eval_indist.json
│  │  ├─ pilot_merged/pilot_train_pairs.json         학습용 질문 23,045쌍 (13라운드 병합)
│  │  └─ eval_indist_merged/pilot_train_pairs.json   분포 내 평가용 질문 5,770쌍 (6라운드 병합)
│  └─ data_raw/                   LLM 세션 원본 19라운드 (prompts/responses 포함, 로컬 전용, 커밋 안 함)
└─ docs/                          위 문서 5개
```

## 실행

각 스크립트를 어디서 어떻게 돌리는지는 [SETUP.md](docs/SETUP.md)를 따라가면 된다.

## 테스트

`datasets/`에서 실행 (venv에 pytest 설치되어 있어야 함, `requirements.txt` 참고):

```
.venv\Scripts\python -m pytest tests/ -v
```
