"""
Export semantic_dict_<dim>d.pkl into a lightweight binary (.bin) and text (.txt)
format so that the C SQL post-processor can load and query embeddings with zero dependencies.
"""

import argparse
import os
import pickle
import struct


def export_embeddings(dim=300):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_dir = os.path.join(base_dir, "data")
    pkl_path = os.path.join(data_dir, f"semantic_dict_{dim}d.pkl")
    bin_path = os.path.join(data_dir, f"semantic_dict_{dim}d.bin")
    txt_path = os.path.join(data_dir, f"semantic_dict_{dim}d.txt")

    if not os.path.exists(pkl_path):
        print(f"[Error] Source file not found: {pkl_path}")
        return False

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data.get("embeddings", {})
    actual_dim = data.get("dim", dim)
    word_count = len(embeddings)

    print(f"Exporting {word_count} words (dim={actual_dim})...")

    # 1. Export binary format (.bin)
    # Header: [Magic: 4 bytes "SEMA"] [dim: uint32] [count: uint32]
    # Each entry: [word_len: uint8] [word: utf-8 bytes] [vector: float32 * dim]
    with open(bin_path, "wb") as f_bin:
        f_bin.write(b"SEMA")
        f_bin.write(struct.pack("<II", actual_dim, word_count))
        for word, vec in embeddings.items():
            encoded = word.encode("utf-8")
            if len(encoded) > 255:
                encoded = encoded[:255]
            f_bin.write(struct.pack("<B", len(encoded)))
            f_bin.write(encoded)
            f_bin.write(struct.pack(f"<{actual_dim}f", *vec))

    print(f" -> Binary format saved to: {bin_path}")

    # 2. Export text format (.txt, standard GloVe format)
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        for word, vec in embeddings.items():
            vec_str = " ".join(f"{val:.6f}" for val in vec)
            f_txt.write(f"{word} {vec_str}\n")

    print(f" -> Text format saved to: {txt_path}")
    print("[Success] All formats exported successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Python semantic_dict.pkl to C-compatible formats (.bin and .txt)"
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        choices=[50, 100, 200, 300],
        default=300,
        help="GloVe embedding dimension (default: 300)",
    )
    args = parser.parse_args()
    export_embeddings(args.dim)
