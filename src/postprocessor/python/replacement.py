import difflib
import os
import pickle
import numpy as np
import glob

# ==========================================
# [GLOBAL CACHE]
# 의미론적 교정에 사용되는 딕셔너리(캐시)
# ==========================================
_semantic_data = {}


def get_semantic_data(dim=300):
    """
    주어진 차원(dim)에 해당하는 GloVe 압축 사전(.pkl)을 로드합니다.
    이미 로드된 적이 있다면 전역 변수에서 캐시된 데이터를 반환하여
    메모리 낭비와 로딩 시간을 절약합니다.
    """
    global _semantic_data
    if dim not in _semantic_data:
        data_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            ),
            "data",
        )
        target_file = os.path.join(data_dir, f"semantic_dict_{dim}d.pkl")

        if os.path.exists(target_file):
            with open(target_file, "rb") as f:
                _semantic_data[dim] = pickle.load(f)
        else:
            # 지정된 차원의 사전이 없을 경우 사용 가능한 다른 사전을 로드합니다.
            pkl_files = glob.glob(os.path.join(data_dir, "semantic_dict_*d.pkl"))
            if pkl_files:
                pkl_files.sort(reverse=True)
                with open(pkl_files[0], "rb") as f:
                    _semantic_data[dim] = pickle.load(f)
            else:
                _semantic_data[dim] = {}
    return _semantic_data[dim]


# ==========================================
# 1. Typo (편집 거리 기반 오타 교정)
# 문자열의 형태적 삽입/삭제/변경 횟수를 기준으로
# 단순 오타를 빠르게 교정하는 알고리즘
# ==========================================
def typo_correction(word, dictionary):
    """
    difflib 모듈을 활용하여 입력 단어와 가장 형태가 유사한 단어를 반환합니다.
    임계값(cutoff) 이상인 단어가 없으면 원본 단어를 그대로 반환합니다.
    """
    matches = difflib.get_close_matches(word, dictionary, n=1, cutoff=0.7)
    return matches[0] if matches else word


# ==========================================
# 2. N-gram (부분 문자열 패턴 교정)
# 단어를 N개의 길이로 쪼갠 집합 간의 Jaccard 유사도를
# 계산하여 변형이 심하거나 긴 단어의 패턴을 매칭하는 알고리즘
# ==========================================
def get_ngrams(word, n=2):
    """
    주어진 단어를 n 길이의 문자열 묶음(집합)으로 분해합니다.
    """
    return set([word[i : i + n] for i in range(len(word) - n + 1)])


def ngram_correction(word, dictionary, n=2, cutoff=0.3):
    """
    단어의 n-gram 집합과 도메인 사전 내 단어의 n-gram 집합 간
    교집합/합집합 비율(Jaccard 유사도)을 계산하여 교정합니다.
    단어가 n보다 짧은 경우 Typo 알고리즘으로 폴백합니다.
    """
    if len(word) < n:
        return typo_correction(word, dictionary)

    word_ngrams = get_ngrams(word, n)
    best_match = word
    best_score = 0.0

    for dict_word in dictionary:
        if len(dict_word) < n:
            continue
        dict_ngrams = get_ngrams(dict_word, n)
        intersection = word_ngrams.intersection(dict_ngrams)
        union = word_ngrams.union(dict_ngrams)
        score = len(intersection) / len(union) if union else 0

        if score > best_score and score >= cutoff:
            best_score = score
            best_match = dict_word

    return best_match


# ==========================================
# 3. Semantic (의미론적 임베딩 기반 교정)
# 형태가 아예 달라도(예: client -> customer)
# 코사인 유사도가 높은 의미상 일치하는 단어로 치환하는 알고리즘
# ==========================================
def split_snake_case(word):
    """
    'user_name'과 같은 스네이크 케이스 문자열을 '_' 기준으로 잘라 리스트로 반환합니다.
    """
    return word.lower().split("_")


def get_mean_pooled_vector(word, embeddings_dict, dim):
    """
    스네이크 케이스로 쪼개진 각 단어의 임베딩 벡터를 구한 뒤,
    합산 후 단어 개수로 나누어 평균 풀링(Mean Pooling)된 최종 벡터를 반환합니다.
    사전에 없는(OOV) 단어는 무시되며, 모두 없으면 영벡터(Zero Vector)를 반환합니다.
    """
    parts = split_snake_case(word)
    vectors = []
    for part in parts:
        if part in embeddings_dict:
            vectors.append(embeddings_dict[part])

    if not vectors:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


def cosine_similarity(vec1, vec2):
    """
    두 벡터 간의 코사인 유사도를 계산합니다. (Numpy 활용)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def semantic_correction(word, dictionary, dim=None, cutoff=0.0):
    """
    입력 단어의 평균 풀링 벡터와 사전 타겟 단어의 평균 풀링 벡터 간
    코사인 유사도를 계산하여 가장 의미론적으로 적합한 단어를 반환합니다.
    """
    if dim is None:
        raise ValueError("dim is required for semantic correction")
    semantic_data = get_semantic_data(dim)
    if not semantic_data or "embeddings" not in semantic_data:
        return word

    embeddings = semantic_data["embeddings"]
    actual_dim = semantic_data["dim"]

    word_vec = get_mean_pooled_vector(word, embeddings, actual_dim)
    if np.linalg.norm(word_vec) == 0:
        return word

    best_match = word
    best_score = 0.0

    # 캐시를 생성했을 당시 존재하던 타겟만 유효하게 검사
    valid_targets = [t for t in dictionary if t in semantic_data["targets"]]
    if not valid_targets:
        valid_targets = semantic_data["targets"]

    for target in valid_targets:
        target_vec = get_mean_pooled_vector(target, embeddings, actual_dim)
        if np.linalg.norm(target_vec) == 0:
            continue

        score = cosine_similarity(word_vec, target_vec)
        if score > best_score and score >= cutoff:
            best_score = score
            best_match = target

    return best_match


# ==========================================
# 파이프라인(Chaining) 인터페이스
# ==========================================
def apply_correction(word, dictionary, strategy, dim, threshold=0.0):
    """
    단일 전략을 선택하여 교정 함수를 호출하는 라우터 역할을 수행합니다.
    """
    if strategy == "typo":
        return typo_correction(word, dictionary)
    elif strategy == "ngram":
        return ngram_correction(word, dictionary)
    elif strategy == "semantic":
        return semantic_correction(word, dictionary, dim=dim, cutoff=threshold)
    return word


def correct_identifiers(
    identifiers, dictionary, strategy="typo,ngram,semantic", dim=None, threshold=0.0
):
    """
    AST에서 추출된 식별자들을 대상으로 전략 기반 교정을 적용합니다.
    여러 전략이 콤마로 나열된 경우(예: 'typo,ngram,semantic'),
    가장 앞선 전략부터 순차적으로 시도(Fallback)하여 교정에 성공하면 중단합니다.
    """
    strategies = [s.strip() for s in strategy.split(",")]

    for node in identifiers:
        original_name = node.name
        is_upper = original_name.isupper()
        check_name = original_name.lower()

        corrected_name = check_name

        for strat in strategies:
            new_name = apply_correction(corrected_name, dictionary, strat, dim, threshold)
            if new_name != corrected_name:
                corrected_name = new_name
                break

        # 교정이 일어났다면 원래 대소문자 속성에 맞추어 변환 후 저장
        if corrected_name != check_name:
            final_name = corrected_name.upper() if is_upper else corrected_name
            node.set("this", final_name)
