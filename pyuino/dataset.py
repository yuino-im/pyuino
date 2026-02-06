import logging
import torch
import random
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from sudachipy import tokenizer
from sudachipy import dictionary
from datasets import load_dataset
from .dictionary import YuinoDicPosId


dataset_path = "range3/cc100-ja"

class YuinoDatasets(Dataset):
    def __init__(self, teacher_model: AutoModel, teacher_tokenizer: AutoTokenizer, cache_dir: str, data_len=0, data_len_per=0.01):
        self._dataset = load_dataset(dataset_path, cache_dir=cache_dir)
        self._sudachi_tokenizer = dictionary.Dictionary(dict="full").create()
        self._embeddings = teacher_model.get_input_embeddings()
        self._tokenizer = teacher_tokenizer
        self._bos_emb = self._embeddings(self._tokenizer.encode("[CLS]", add_special_tokens=False, return_tensors="pt")).squeeze().tolist()
        self._eos_emb = self._embeddings(self._tokenizer.encode("[SEP]", add_special_tokens=False, return_tensors="pt")).squeeze().tolist()
        self._pos_ids = YuinoDicPosId()

        if data_len <= 0:
            data_len = int(self._dataset.num_rows["train"] * data_len_per)
        idx = list(range(self._dataset.num_rows["train"]))
        self._data_idx = random.sample(idx, data_len)
        logging.info("Data read size= %d" % data_len)

    def __len__(self):
        return len(self._data_idx)

    def __getitem__(self, idx):
        i = 0
        text = self._dataset.data["train"]["text"][self._data_idx[idx]].as_py()
        while len(text) < 2:
            i += 1
            text = self._dataset.data["train"]["text"][self._data_idx[idx] + i].as_py()

        text = text.strip()
        token = self._sudachi_tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.C)

        # add BOS
        l_ary = [self._bos_emb]
        p_ary = [self._pos_ids.bos_id]

        # add tables
        for m in token:
            try:
                t = self._tokenizer.encode(m.surface(), add_special_tokens=False, return_tensors="pt")
                l_ary.append(torch.mean(self._embeddings(t), dim=1).squeeze().tolist())
                p_ary.append(self._pos_ids.get_pos_id(m.part_of_speech()))

            except RuntimeError:
                continue

        # add EOS
        l_ary.append(self._eos_emb)
        p_ary.append(self._pos_ids.eos_id)

        return {
            "labels": torch.tensor(l_ary).detach(),
            "inputs_poss": torch.tensor(p_ary).detach(),
            "attention_mask": torch.tensor([1 for _ in range(len(p_ary))]).detach(),
        }
