import sys
import os
import json

# 프로젝트 루트 디렉터리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import dataset as ds
from src.data.dataset import BOS_ID, EOS_ID, SEP_ID, PAD_ID, TextToSQLDataset, collate_fn


# encode_pair가 [<bos> question <sep> sql <eos>] 구조를 정확히 만드는지, <sep>이 한 번만 들어가는지 검증
def test_encode_pair_structure():
    # merges=[] 를 줘도 바이트 레벨 인코딩으로 동작하므로 학습 없이 구조만 검증 가능
    ids = ds.encode_pair("select name", "SELECT name", merges=[])

    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert SEP_ID in ids
    assert ids.count(SEP_ID) == 1


# tokenize_pairs가 계산한 sql_start 인덱스가 실제로 <sep> 바로 다음(SQL 시작 지점)을 가리키는지 검증
def test_tokenize_pairs_sql_start():
    pairs = [{"question": "hi", "sql": "SELECT 1"}]
    examples = ds.tokenize_pairs(pairs, merges=[])
    ex = examples[0]

    # sql_start 바로 앞은 <sep>, sql_start부터는 SQL 쪽 토큰이어야 함
    assert ex["ids"][ex["sql_start"] - 1] == SEP_ID
    assert ex["ids"][ex["sql_start"]] != SEP_ID
    assert ex["ids"][-1] == EOS_ID


# TextToSQLDataset이 examples 리스트를 그대로 인덱싱/길이 반환하는 Dataset 프로토콜을 만족하는지 검증
def test_dataset_len_and_getitem():
    examples = [
        {"ids": [1, 2, 3], "sql_start": 1},
        {"ids": [4, 5], "sql_start": 1},
    ]
    dataset = TextToSQLDataset(examples)

    assert len(dataset) == 2
    assert dataset[0] == examples[0]
    assert dataset[1] == examples[1]


# collate_fn이 배치 내 최대 길이에 맞춰 PAD_ID로 패딩하고, next-token-prediction용 input/target shift를 올바르게 만드는지 검증
def test_collate_fn_padding_and_shift():
    batch = [
        {"ids": [10, 11, 12, 13], "sql_start": 2},  # 길이 4 (배치 내 최대 길이)
        {"ids": [20, 21], "sql_start": 1},           # 짧은 쪽 -> 패딩 필요
    ]
    out = collate_fn(batch)

    # max_len=4 이므로 shift 이후 길이는 3
    assert out["input_ids"].shape == (2, 3)
    assert out["target_ids"].shape == (2, 3)
    assert out["loss_mask"].shape == (2, 3)

    assert out["input_ids"][0].tolist() == [10, 11, 12]
    assert out["target_ids"][0].tolist() == [11, 12, 13]

    assert out["input_ids"][1].tolist() == [20, 21, PAD_ID]
    assert out["target_ids"][1].tolist() == [21, PAD_ID, PAD_ID]


# collate_fn의 loss_mask가 SQL 토큰+<eos> 구간에만 True이고, 패딩 위치는 항상 False인지 검증
def test_collate_fn_loss_mask_only_covers_sql_and_eos():
    batch = [
        {"ids": [10, 11, 12, 13], "sql_start": 2},
        {"ids": [20, 21], "sql_start": 1},
    ]
    out = collate_fn(batch)

    # 각 행에서 loss_mask가 True인 개수 == (원본 길이 - sql_start), 즉 SQL 토큰 + <eos> 개수
    for i, ex in enumerate(batch):
        expected = len(ex["ids"]) - ex["sql_start"]
        assert out["loss_mask"][i].sum().item() == expected

    # 패딩 위치는 항상 loss_mask=False
    assert out["loss_mask"][1][1:].tolist() == [False, False]


# load_pairs가 glob으로 매칭된 여러 파일의 JSON을 경로 정렬 순서대로 병합해서 반환하는지 검증
def test_load_pairs_merges_and_sorts_multiple_files(tmp_path):
    (tmp_path / "pilot1").mkdir()
    (tmp_path / "pilot2").mkdir()
    with open(tmp_path / "pilot1" / "pairs.json", "w", encoding="utf-8") as f:
        json.dump([{"question": "q1", "sql": "s1"}], f)
    with open(tmp_path / "pilot2" / "pairs.json", "w", encoding="utf-8") as f:
        json.dump([{"question": "q2", "sql": "s2"}], f)

    pairs = ds.load_pairs(tmp_path, "pilot*/pairs.json")

    # pilot1, pilot2 순서(경로 정렬)로 병합되어야 함
    assert pairs == [
        {"question": "q1", "sql": "s1"},
        {"question": "q2", "sql": "s2"},
    ]


# load_pairs가 glob에 매칭되는 파일이 하나도 없을 때 예외 없이 빈 리스트를 반환하는지 검증
def test_load_pairs_no_match_returns_empty_list(tmp_path):
    pairs = ds.load_pairs(tmp_path, "nonexistent*/pairs.json")
    assert pairs == []
