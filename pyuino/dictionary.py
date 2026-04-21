import os
import csv
import jaconv
import torch
import marisa_trie
from typing import Optional, List
from logging import getLogger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .model import YuinoModel
from .pb import YuinoWord, YuinoPos, YuinoDic


def _get_tqdm_bar(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=",")
        bar = tqdm(total=sum(1 for _ in reader))
    return bar


def build_dictionary(teacher_model: AutoModel, teacher_tokenizer: AutoTokenizer):
    logger = getLogger("YuinoDictionaryBuilder")
    dic_csv_files = [
        "small_lex.csv",
        "core_lex.csv",
        "notcore_lex.csv"
    ]
    model = YuinoModel.from_pretrained("YuinoLM")
    pos_id = YuinoDicPosId()
    anya_words = YuinoDic()
    poss = []
    words = []

    embeddings = teacher_model.get_input_embeddings()

    # reg pos_id
    for i in range(pos_id.pos_id_size):
        pos = YuinoPos()
        pos.id = i
        pos.vec = model.get_pos_id(torch.tensor(i))
        poss.append(pos)
    anya_words.poss.extend(poss)

    # add BOS (id=0)
    word = YuinoWord()
    word.surface = "[CLS]"
    word.read = "[CLS]"
    word.pos = pos_id.bos_id
    vec = model.get_uint_id(embeddings(teacher_tokenizer.encode("[CLS]", add_special_tokens=False, return_tensors="pt")))
    word.vector = vec
    words.append(word)

    for csv_file in dic_csv_files:
        read_file_path = os.path.join("./dict", csv_file)
        logger.info(" reading... %s" % read_file_path)
        bar = _get_tqdm_bar(read_file_path)
        with open(read_file_path) as f:
            reader = csv.reader(f, delimiter=",")
            try:
                for line in reader:
                    try:
                        word = YuinoWord()
                        word.surface = line[0]
                        word.read = jaconv.kata2hira(line[11])
                        word.pos = pos_id.get_pos_id((line[5], line[6], line[7], line[8], line[9], line[10]))
                        e = embeddings(teacher_tokenizer.encode(line[0], return_tensors="pt", add_special_tokens=False))
                        word.vector = model.get_uint_id(torch.mean(e, dim=1).unsqueeze(0))

                        words.append(word)

                    except RuntimeError as e:
                        print(word.surface, " : not vectorized  ", e)

                    bar.update(1)

            except UnicodeDecodeError as e:
                logger.warning(e)

    write_file_path = os.path.join("./YuinoLM", "yuino_dict.pb")
    logger.info("all read OK! and writing -> %s" % write_file_path)
    anya_words.words.extend(words)
    with open(write_file_path, "wb") as f:
        f.write(anya_words.SerializeToString())


class YuinoDicPosId:
    def __init__(self, pos_id_file="./YuinoLM/pos_id.csv"):
        self._pos_ids = ["PAD"]

        with open(pos_id_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                self._pos_ids.append(row[0] + "." + row[1] + "." + row[2] + "." + row[3] + "." + row[4] + "." + row[5])

        # tail add
        self._pos_ids.append("BOS.*.*.*.*.*")
        self._pos_ids.append("EOS.*.*.*.*.*")
        self._pos_ids.append("UNK.*.*.*.*.*")

    @property
    def pos_id_size(self):
        return len(self._pos_ids)

    @property
    def bos_id(self):
        return self.get_pos_id(("BOS", "*", "*", "*", "*", "*"))

    @property
    def eos_id(self):
        return self.get_pos_id(("EOS", "*", "*", "*", "*", "*"))

    def get_pos_id(self, inputs: tuple) -> int:
        return self._pos_ids.index(inputs[0] + "." + inputs[1] + "." + inputs[2] + "." + inputs[3] + "." + inputs[4] + "." + inputs[5])


class YuinoDictionary:
    def __init__(self, model_path: Optional[str]=None, num_bits=64):
        self._pos_id = YuinoDicPosId()
        self._pos_emb = torch.eye(self._pos_id.pos_id_size)
        self._logger = getLogger("YuinoDictionary")

        dict_file_path = os.path.join(model_path, "yuino_dict.pb")
        yuino_words = YuinoDic()
        with open(dict_file_path, "rb") as f:
            data = f.read()
            yuino_words.ParseFromString(data)

        trie_key = []
        trie_val = []
        self._words = []
        self._pos_vec = []

        # add words
        for i, word in enumerate(yuino_words.words):
            trie_key.append(word.read)
            trie_val.append((i,))
            self._words.append(word)

        for pos in yuino_words.poss:
            self._pos_vec.append(pos.vec)

        # build trie
        self._trie = marisa_trie.RecordTrie("<L", zip(trie_key, trie_val))
        self._shifts = torch.arange(num_bits - 1, -1, -1)

    @property
    def pos_id_size(self):
        return self._pos_id.pos_id_size

    @property
    def bos_id(self):
        return 0

    def build_word_tree(self, in_text):
        in_len = len(in_text) + 1
        nodes_set = [[] for _ in range(in_len)]
        nodes_set[0] = ["[CLS]"]
        for i in range(in_len):
            for prefix in self._trie.prefixes(in_text[i:]):
                nodes_set[i + len(prefix)].append(prefix)
        return nodes_set

    def gets(self, ym):
        return list(set([wid[0] for wid in self._trie[ym]]))

    def surface(self, wid):
        return self._words[wid].surface

    def pos(self, wid):
        return self._words[wid].pos

    def embed(self, words: List[int]):
        wt = []
        for wid in words:
            w_vec = (self._words[wid].vector >> self._shifts) & 1
            p_vec = (self._pos_vec[self.pos(wid)] >> self._shifts) & 1
            emb = torch.cat([w_vec, p_vec]).float()
            wt.append(emb.unsqueeze(0))
        return torch.cat(wt).unsqueeze(0)
