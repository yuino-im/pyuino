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


class YuinoDicPosId:
    def __init__(self):
        self._pos_ids = [
            "PAD",
            "名詞.普通名詞.一般.*",
            "名詞.普通名詞.サ変可能.*",
            "名詞.普通名詞.形状詞可能.*",
            "名詞.普通名詞.サ変形状詞可能.*",
            "名詞.普通名詞.副詞可能.*",
            "名詞.普通名詞.助数詞可能.*",
            "名詞.固有名詞.一般.*",
            "名詞.固有名詞.人名.一般",
            "名詞.固有名詞.人名.姓",
            "名詞.固有名詞.人名.名",
            "名詞.固有名詞.地名.一般",
            "名詞.固有名詞.地名.国",
            "名詞.数詞.*.*",
            "名詞.助動詞語幹.*.*",
            "代名詞.*.*.*",
            "形状詞.一般.*.*",
            "形状詞.タリ.*.*",
            "形状詞.助動詞語幹.*.*",
            "連体詞.*.*.*",
            "副詞.*.*.*",
            "接続詞.*.*.*",
            "感動詞.一般.*.*",
            "感動詞.フィラー.*.*",
            "動詞.一般.*.*",
            "動詞.非自立可能.*.*",
            "形容詞.一般.*.*",
            "形容詞.非自立可能.*.*",
            "助動詞.*.*.*",
            "助詞.格助詞.*.*",
            "助詞.副助詞.*.*",
            "助詞.係助詞.*.*",
            "助詞.接続助詞.*.*",
            "助詞.終助詞.*.*",
            "助詞.準体助詞.*.*",
            "接頭辞.*.*.*",
            "接尾辞.名詞的.一般.*",
            "接尾辞.名詞的.サ変可能.*",
            "接尾辞.名詞的.形状詞可能.*",
            "接尾辞.名詞的.副詞可能.*",
            "接尾辞.名詞的.助数詞.*",
            "接尾辞.形状詞的.*.*",
            "接尾辞.動詞的.*.*",
            "接尾辞.形容詞的.*.*",
            "記号.一般.*.*",
            "記号.文字.*.*",
            "補助記号.一般.*.*",
            "補助記号.句点.*.*",
            "補助記号.読点.*.*",
            "補助記号.括弧開.*.*",
            "補助記号.括弧閉.*.*",
            "補助記号.ＡＡ.一般.*",
            "補助記号.ＡＡ.顔文字.*",
            "空白.*.*.*",
            "BOS.*.*.*",
            "EOS.*.*.*",
            "UNK.*.*.*"
        ]

    @property
    def pos_id_size(self):
        return len(self._pos_ids)

    @property
    def bos_id(self):
        return self.get_pos_id(("BOS", "*", "*", "*"))

    @property
    def eos_id(self):
        return self.get_pos_id(("EOS", "*", "*", "*"))

    def get_pos_id(self, inputs: tuple) -> int:
        return self._pos_ids.index(inputs[0] + "." + inputs[1] + "." + inputs[2] + "." + inputs[3])


class YuinoDictionary:
    def __init__(self, model_path: Optional[str]=None, device="cpu", num_bits=32):
        self._dict_path = model_path
        self._pos_id = YuinoDicPosId()
        self._pos_emb = torch.eye(self._pos_id.pos_id_size)
        self._shifts = torch.arange(num_bits - 1, -1, -1)
        self._logger = getLogger("YuinoDictionary")

        self._device = device
        self._model = YuinoModel.from_pretrained(model_path).to(self._device)
        self._loss_func = torch.nn.BCEWithLogitsLoss()

        dict_file_path = os.path.join(self._dict_path, "yuino_dict.pb")
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
        self._first_embed = self.predict([self.bos_id])

    @property
    def pos_id_size(self):
        return self._pos_id.pos_id_size

    @property
    def bos_id(self):
        return 0

    @property
    def first_embed(self):
        return self._first_embed

    def embed(self, words: List[int]):
        wt = []
        for wid in words:
            w_vec = (self._words[wid].vector >> self._shifts) & 1
            p_vec = (self._pos_vec[self.pos(wid)] >> self._shifts) & 1
            emb = torch.cat([w_vec, p_vec]).float()
            wt.append(emb.unsqueeze(0))
        return torch.cat(wt).unsqueeze(0).to(self._device)

    def predict(self, words: List[int]):
        wt = self.embed(words)
        y = self._model(inputs_embeds=wt)
        return y.logits[:, -1, :]

    def build_word_tree(self, in_text):
        word_set = [[] for _ in range(len(in_text))]
        for i in range(len(in_text)):
            for prefix in self._trie.prefixes(in_text[i:]):
                word_set[i].append(prefix)
        return word_set

    def gets(self, ym):
        return list(set([wid[0] for wid in self._trie[ym]]))

    def loss(self, y, y_hat):
        return self._loss_func(y, y_hat)

    def surface(self, wid):
        return self._words[wid].surface

    def pos(self, wid):
        return self._words[wid].pos

    @staticmethod
    def build(teacher_model: AutoModel, teacher_tokenizer: AutoTokenizer):
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
                            word.pos = pos_id.get_pos_id((line[5], line[6], line[7], line[8]))
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
