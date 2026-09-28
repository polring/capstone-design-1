# config.py

# 현재 학습 단계 — 단계를 바꿀 때는 이 값만 수정한다
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
FFN_DIM = HIDDEN_DIM * 3

# Training
BATCH_SIZE = 32       # DataLoader 배치 크기

# Paths (저장소 루트 기준 상대 경로)
DATA_DIR = "data"                     # 커밋되는 데이터 산출물 폴더
HOLDOUT_PATH = "data/holdout.json"    # 미등장 평가용 엔티티
BPE_MERGES_PATH = "bpe_merges.json"   # BPE 학습 산출물 (커밋 안 함)
