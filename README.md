# 🤖 Text-to-SQL Transformer LLM (Scratch Build)

4인 캡스톤 디자인 프로젝트: 밑바닥(Scratch)부터 구현하는 경량 Text-to-SQL Transformer LLM 저장소입니다.

자연어 질문(예: `how many mouse units are left in stock?`)을 입력받아 SQLite SQL 쿼리(`SELECT stock FROM items WHERE item_name = 'mouse'`)로 변환하는 언어 모델을 구현합니다. 모델, 토크나이저, 학습 루프, 추론을 모두 직접 구현하고 사전학습 가중치는 쓰지 않습니다.

---

## 🚀 바로 써 보기

학습된 배포 모델(`models/stage3/`, 이전 단계는 `models/stage1/`·`models/stage2/`)이 저장소에 포함되어 있어 학습 없이 바로 실행할 수 있습니다.
기본은 3단계 모델이고, 이전 단계 모델은 `--stage 1` 또는 `--stage 2`를 붙입니다.

```bash
pip install -r requirements.txt
python -m src.run --input "what city does ashley live in?"   # 질문 → SQL → DB 실행 결과
python -m src.run --input "the 3 cheapest kitchen items with more than 10 in stock"
python -m src.run --stage 1 --input "what city does ashley live in?"   # 1단계 모델
python -m src.run                                            # 대화형
python -m src.run --evaluation                               # 평가셋 전부 채점
python -m src.demo all                                       # 예시 하나로 파이프라인 각 단계의 입력 → 출력 보기
pytest tests/ -v                                             # 단위 테스트
```

> **참고**: PyTorch CUDA 빌드 인덱스(`--extra-index-url`)가 `requirements.txt`에 포함되어 있습니다. 추론·평가는 CPU로도 됩니다(`--cpu`).
> 모든 명령은 저장소 루트에서 실행합니다. 학습·데이터 생성·인자 설명은 [docs/SETUP.md](docs/SETUP.md)에 있습니다.
> C 추론 엔진과 SQL 후처리기는 별도 저장소 [capstone-sql-cli](https://github.com/polring/capstone-sql-cli)에 있고,
> 이 저장소는 GGUF 모델을 GitHub Release로 배포합니다([docs/exporter-loader.md](docs/exporter-loader.md)).

---

## 📁 프로젝트 구조

```text
capstone-design-1/
├── src/                    # 코드. 실행 명령(config, train, release, run, demo, exporter)은 src/ 바로 아래
│   ├── models/             # Transformer 모델 (임베딩·RoPE, 디코더 블록, 조립)
│   ├── tokenizer/          # BPE 토크나이저
│   ├── inference/          # 학습된 모델 불러오기, SQL 생성
│   ├── evaluation/         # 채점·평가셋 (단계 공통)
│   ├── data/               # 공통 DB·스키마·증강 + stage1/, stage2/, stage3/ 전용 SQL·질문 생성
│   └── postprocessor/      # SQL 후처리 (python/)
├── data/                   # 커밋되는 데이터
│   ├── stage<N>/           # 단계별: db, sql, train, eval, tokenizer, composer 등
│   └── common/wordnet/     # 이름 교체용 영단어 목록
├── models/stage<N>/        # 배포 모델 (model.pt, tokenizer.json, model_info.json)
├── licenses/               # 외부 데이터·라이브러리 라이선스 원본
├── tests/                  # pytest — src/ 와 같은 구조
└── docs/                   # 설계서, 단계별 보고서, 라벨 규칙, 실행 방법, 협업 규칙

```

파일 단위 설명은 [docs/SETUP.md](docs/SETUP.md) 3절에 있습니다.

커밋하지 않는 폴더: `data_raw/`(질문 생성 라운드·중간 산출물), `runs/`(학습·평가·추론 결과),
`experiments/`(단계 전환 전 실험 코드·데이터·체크포인트).

---

## 📜 협업 규칙 및 문서
- 팀 협업 및 Git 커밋/PR 규칙은 [docs/CONVENTION.md](docs/CONVENTION.md)를 확인하세요.
- 데이터셋 구조와 모델 아키텍처 설계는 [docs/text-to-sql-llm-project-plan.md](docs/text-to-sql-llm-project-plan.md)를 참고하세요.
- 현재 진행 상황과 결과는 [1단계 보고서](docs/text-to-sql-stage1-report.md)·[2단계 보고서](docs/text-to-sql-stage2-report.md)·[3단계 보고서](docs/text-to-sql-stage3-report.md), 실행·재현 방법은 [docs/SETUP.md](docs/SETUP.md)에 있습니다.

3단계도 `python -m src.train --stage 3`, `python -m src.run --stage 3` 등 공통 실행 코드를 사용합니다.
[3단계 코드 구성](docs/text-to-sql-stage3-report.md#code-layout)을 참고하세요.

3단계 배포 모델은 `models/stage3/`의 15.7M 모델입니다. 토크나이저와 최종 전처리 사전을 함께 제공합니다.

WordNet 라이선스 원본은 [licenses/WORDNET_LICENSE.txt](licenses/WORDNET_LICENSE.txt)에서 관리하며,
전처리 사전을 별도 배포할 때는 라이선스를 배포 파일에 함께 포함합니다.

```sh
python -m src.run --stage 3 --wordnet-map models/stage3/normalizer.json --input "record count of transactions"
```
