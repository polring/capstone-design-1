# config.py

# Hyperparameters for a 50M-100M parameter mini LLM
VOCAB_SIZE = 32000    # BPE Tokenizer 단어장 크기 (16k ~ 32k)
HIDDEN_DIM = 512      # 임베딩 및 은닉층 차원
SEQ_LEN = 1024        # 최대 문맥 길이 (Context Length)
NUM_HEADS = 8         # 멀티 헤드 어텐션의 헤드 개수
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 각 헤드의 차원

NUM_LAYERS = 6              # Transformer 블록 개수
FFN_DIM = HIDDEN_DIM * 3 