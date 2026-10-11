# 영단어 목록 (WordNet 3.0 표제어)

이름 교체 학습(`src/data/stage1/name_swap.py`)이 가짜 이름에서 실제 영단어를 걸러 낼 때 쓴다. `config.WORDLIST_PATH`.

- `english_words.txt`: WordNet 3.0의 `index.noun/verb/adj/adv` 표제어 중 알파벳으로만 된 것을 소문자로, 한 줄에
  한 단어씩 정렬해 저장했다(77,503개). NLTK WordNet 데이터(`wordnet.zip`)에서 `name_swap.load_english_words()`로
  읽은 집합과 같다.
- `LICENSE`: WordNet 3.0 라이선스 원문. 재배포 시 함께 두어야 한다.
