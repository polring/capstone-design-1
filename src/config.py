# config.py — 모델 구조, 학습 설정, 데이터·모델 저장 경로
# 표준 라이브러리만 쓴다 (torch 없이 도는 질문 생성 스크립트도 import 한다)

from dataclasses import dataclass
from pathlib import Path

# 현재 학습 단계 — 단계를 바꿀 때는 이 값만 수정한다 (모델 구조와 아래 경로가 함께 바뀐다)
STAGE = 1

# 단계별 모델 구조 (계획서 3-2 단계별 자원 계획, 5 = 상한)
STAGE_PRESETS = {
    1: dict(NUM_LAYERS=6,  HIDDEN_DIM=256, NUM_HEADS=4,  VOCAB_SIZE=1024, SEQ_LEN=128),  # 5.02M
    2: dict(NUM_LAYERS=8,  HIDDEN_DIM=384, NUM_HEADS=6,  VOCAB_SIZE=2048, SEQ_LEN=128),  # 15.0M
    3: dict(NUM_LAYERS=8,  HIDDEN_DIM=512, NUM_HEADS=8,  VOCAB_SIZE=4096, SEQ_LEN=256),  # 27.4M
    4: dict(NUM_LAYERS=12, HIDDEN_DIM=512, NUM_HEADS=8,  VOCAB_SIZE=8192, SEQ_LEN=512),  # 42.2M
    5: dict(NUM_LAYERS=12, HIDDEN_DIM=768, NUM_HEADS=12, VOCAB_SIZE=8192, SEQ_LEN=512),  # 91.6M
}

_preset = STAGE_PRESETS[STAGE]
VOCAB_SIZE = _preset["VOCAB_SIZE"]    # BPE Tokenizer 단어장 크기
HIDDEN_DIM = _preset["HIDDEN_DIM"]    # 임베딩 및 은닉층 차원
SEQ_LEN = _preset["SEQ_LEN"]          # 최대 문맥 길이 (Context Length)
NUM_HEADS = _preset["NUM_HEADS"]      # 멀티 헤드 어텐션의 헤드 개수
HEAD_DIM = HIDDEN_DIM // NUM_HEADS    # 각 헤드의 차원 (모든 단계 64)
NUM_LAYERS = _preset["NUM_LAYERS"]    # 디코더 블록 개수
FFN_DIM = round(HIDDEN_DIM * 8 / 3)  # SwiGLU(행렬 3개)가 계획서의 GELU FFN(4d, 행렬 2개, 8d²)과 같은 파라미터 예산이 되는 크기

# Training
BATCH_SIZE = 32       # DataLoader 배치 크기

# Paths (저장소 루트 기준 상대 경로). 역할은 폴더 위치로 정한다:
# train/ 안의 *.json 은 전부 학습 데이터·BPE 코퍼스, eval/ 아래 폴더 하나가 평가셋 하나다.
PAIRS_FILE = "pairs.json"                          # 평가셋 폴더의 (질문, SQL) 쌍 파일 이름
EVAL_SQL_FILE = "sql.json"                         # 평가셋 폴더의 SQL 메타데이터 파일 이름


@dataclass(frozen=True)
class StagePaths:
    """단계 하나의 데이터·모델 경로. 아래 모듈 상수는 현재 STAGE 의 값이고,
    다른 단계의 경로가 필요하면(예: run.py --stage 2) stage_paths(N) 을 쓴다."""
    stage: int

    @property
    def data_dir(self) -> Path: return Path("data") / f"stage{self.stage}"        # 이 단계의 커밋되는 데이터
    @property
    def db_dir(self) -> Path: return self.data_dir / "db"
    @property
    def db_path(self) -> Path: return self.db_dir / "shop.db"                     # SQLite DB
    @property
    def holdout_path(self) -> Path: return self.db_dir / "holdout.json"           # 미등장 평가용 엔티티 (학습 SQL 에 절대 나오면 안 됨)
    @property
    def sql_dir(self) -> Path: return self.data_dir / "sql"                       # sql_gen 출력
    @property
    def train_dir(self) -> Path: return self.data_dir / "train"                   # 학습 (질문, SQL) 쌍
    @property
    def eval_dir(self) -> Path: return self.data_dir / "eval"                     # 평가셋: <이름>/pairs.json + sql.json
    @property
    def seed_tokens_path(self) -> Path: return self.data_dir / "tokenizer" / "seed_tokens.json"  # BPE seed 토큰 (입력, 사람이 편집)
    @property
    def tokenizer_path(self) -> Path: return self.data_dir / "tokenizer" / "tokenizer.json"      # BPE 학습 결과
    @property
    def model_dir(self) -> Path: return Path("models") / f"stage{self.stage}"     # 배포용 최종 모델 (커밋)
    @property
    def model_path(self) -> Path: return self.model_dir / "model.pt"              # 같은 폴더에 tokenizer.json, model_info.json
    @property
    def raw_dir(self) -> Path: return Path("data_raw") / f"stage{self.stage}"     # 검증 전 라운드·중간 산출물 (커밋 안 함)
    @property
    def runs_dir(self) -> Path: return Path("runs") / f"stage{self.stage}"        # 학습 실행 결과 (커밋 안 함)
    @property
    def release_eval_dir(self) -> Path: return self.runs_dir / "release_eval"     # 배포 모델 평가 결과 (커밋 안 함)
    @property
    def predictions_dir(self) -> Path: return self.runs_dir / "predictions"       # run.py --test 배치 추론 결과 (커밋 안 함)


def stage_paths(stage: int = STAGE) -> StagePaths:
    return StagePaths(stage)


_paths = stage_paths(STAGE)
DATA_DIR = _paths.data_dir
DB_DIR = _paths.db_dir
DB_PATH = _paths.db_path
HOLDOUT_PATH = _paths.holdout_path
SQL_DIR = _paths.sql_dir
SQL_TRAIN_PATH = SQL_DIR / "train.json"            # 학습용 SQL
SQL_EVAL_INDIST_PATH = SQL_DIR / "eval_indist.json"    # 분포 내 평가용 SQL
SQL_EVAL_HOLDOUT_PATH = SQL_DIR / "eval_holdout.json"  # 미등장 값 평가용 SQL (tier 포함)
SQL_REPORT_PATH = SQL_DIR / "gen_report.json"      # SQL 구조별 할당량 보고서
TRAIN_DIR = _paths.train_dir
EVAL_DIR = _paths.eval_dir
SEED_TOKENS_PATH = _paths.seed_tokens_path
TOKENIZER_PATH = _paths.tokenizer_path
MODEL_DIR = _paths.model_dir
MODEL_PATH = _paths.model_path
RAW_DIR = _paths.raw_dir
RUNS_DIR = _paths.runs_dir
RELEASE_EVAL_DIR = _paths.release_eval_dir
PREDICTIONS_DIR = _paths.predictions_dir

WORDLIST_PATH = Path("data") / "wordnet" / "english_words.txt"  # 이름 교체용 영단어 (WordNet 3.0 표제어, 단계 공통)
