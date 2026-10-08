import argparse
import os
import zipfile
import urllib.request
import numpy as np
import pickle
import sys

# schema 모듈을 임포트할 수 있도록 경로를 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schema import load_dictionary_from_sqlite


def download_glove_if_missing(data_dir, dim):
    glove_zip = os.path.join(data_dir, "glove.6B.zip")
    glove_txt = os.path.join(data_dir, f"glove.6B.{dim}d.txt")

    if os.path.exists(glove_txt):
        return glove_txt

    if not os.path.exists(glove_zip):
        print("Downloading GloVe embeddings (this may take a while)...")
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"  # 스탠포드 공식 사이트보다 안정적인 허깅페이스 링크 사용
        urllib.request.urlretrieve(url, glove_zip)
        print("Download complete.")

    print(f"Extracting glove.6B.{dim}d.txt...")
    with zipfile.ZipFile(glove_zip, "r") as zip_ref:
        zip_ref.extract(f"glove.6B.{dim}d.txt", data_dir)

    try:
        os.remove(glove_zip)
        print("Deleted downloaded .zip file to save disk space.")
    except Exception as e:
        print(f"Warning: failed to delete .zip file: {e}")

    return glove_txt


def split_snake_case(word):
    return word.lower().split("_")


def get_mean_pooled_vector(word, embeddings_dict, dim):
    parts = split_snake_case(word)
    vectors = []
    for part in parts:
        if part in embeddings_dict:
            vectors.append(embeddings_dict[part])

    if not vectors:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build lightweight Semantic Dictionary from GloVe using Numpy"
    )
    parser.add_argument(
        "-d", "--dim",
        type=int,
        choices=[50, 100, 200, 300],
        required=True,
        help="GloVe dimension to use (required)",
    )
    parser.add_argument("-b", "--db", type=str, default=None, help="Path to shop.db")
    parser.add_argument(
        "-D", "--data_dir", type=str, default=None, help="Path to data directory"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        required=True,
        help="Cosine similarity threshold (e.g. 0.5) to keep words (required).",
    )

    args = parser.parse_args()

    # 프로젝트 루트 디렉토리를 계산합니다.
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    if args.data_dir is None:
        args.data_dir = os.path.join(base_dir, "data")

    if args.db is None:
        args.db = os.path.join(args.data_dir, "shop.db")

    os.makedirs(args.data_dir, exist_ok=True)

    target_words = load_dictionary_from_sqlite(args.db)
    if not target_words:
        print("No target words found. Check DB path.")
        return

    print(f"Target dictionary size: {len(target_words)}")

    # 선택 사항: 테스트 환경 등에서 다운로드가 너무 느릴 경우를 대비해 수동 다운로드 안내를 출력합니다.
    try:
        glove_txt = download_glove_if_missing(args.data_dir, args.dim)
    except Exception as e:
        print(f"Failed to download/extract GloVe: {e}")
        print("Please manually place glove.6B.zip in the data/ directory.")
        return

    print("Loading GloVe into memory (this may take a minute)...")
    full_glove = {}
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            full_glove[word] = vector

    print("Computing target vectors...")
    target_vectors = {}
    for tw in target_words:
        vec = get_mean_pooled_vector(tw, full_glove, args.dim)
        target_vectors[tw] = vec

    print("Filtering GloVe vocabulary using cosine similarity...")
    target_keys = list(target_vectors.keys())
    target_matrix = np.array([target_vectors[k] for k in target_keys])
    target_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)
    target_norms[target_norms == 0] = 1e-10
    target_matrix = target_matrix / target_norms

    all_words = list(full_glove.keys())
    all_matrix = np.array(list(full_glove.values()))
    all_norms = np.linalg.norm(all_matrix, axis=1, keepdims=True)
    all_norms[all_norms == 0] = 1e-10
    all_matrix = all_matrix / all_norms

    similarities = np.dot(target_matrix, all_matrix.T)

    keep_indices = set()
    for i in range(len(target_keys)):
        # 지정된 유사도 점수 이상의 인덱스만 추출합니다.
        valid_indices = np.where(similarities[i] >= args.threshold)[0]
        keep_indices.update(valid_indices)

    print(
        f"Reduced vocabulary size (Threshold >= {args.threshold}): {len(keep_indices)}"
    )

    reduced_glove = {}
    for idx in keep_indices:
        word = all_words[idx]
        reduced_glove[word] = full_glove[word]

    # 유사도와 관계없이 타겟 단어의 구성 요소(쪼개진 단어)는 무조건 포함되도록 보장합니다.
    for tw in target_words:
        parts = split_snake_case(tw)
        for part in parts:
            if part in full_glove and part not in reduced_glove:
                reduced_glove[part] = full_glove[part]

    output_path = os.path.join(args.data_dir, f"semantic_dict_{args.dim}d.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(
            {"dim": args.dim, "targets": target_keys, "embeddings": reduced_glove}, f
        )

    print(f"Successfully saved reduced dictionary to {output_path}")

    try:
        os.remove(glove_txt)
        print("Deleted extracted .txt file to save disk space.")
    except Exception as e:
        print(f"Warning: failed to delete .txt file: {e}")


if __name__ == "__main__":
    main()
