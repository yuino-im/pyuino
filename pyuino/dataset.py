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
        temp_dataset = load_dataset(dataset_path, cache_dir=cache_dir)
        total_rows = temp_dataset["train"].num_rows

        self._cache_dir = cache_dir
        self._dataset = None
        self._sudachi_tokenizer = None

        self._embeddings = teacher_model.get_input_embeddings()
        for param in self._embeddings.parameters():
            param.requires_grad = False

        self._tokenizer = teacher_tokenizer
        self._bos_emb = self._embeddings(self._tokenizer.encode("[CLS]", add_special_tokens=False, return_tensors="pt")).squeeze().to(dtype=torch.bfloat16).tolist()
        self._eos_emb = self._embeddings(self._tokenizer.encode("[SEP]", add_special_tokens=False, return_tensors="pt")).squeeze().to(dtype=torch.bfloat16).tolist()
        self._pos_ids = YuinoDicPosId()

        if data_len <= 0:
            data_len = int(total_rows * data_len_per)
        idx = list(range(total_rows))
        self._data_idx = random.sample(idx, data_len)
        logging.info("Data read size= %d" % data_len)

    def __len__(self):
        return len(self._data_idx)

    def __getitem__(self, idx):
        if self._dataset is None:
            self._dataset = load_dataset(dataset_path, cache_dir=self._cache_dir)
        if self._sudachi_tokenizer is None:
            self._sudachi_tokenizer = dictionary.Dictionary(dict="full").create()

        i = 0
        text = self._dataset["train"][self._data_idx[idx]]["text"]
        while len(text) < 2:
            i += 1
            text = self._dataset["train"][self._data_idx[idx] + i]["text"]

        text = text.strip()
        token = self._sudachi_tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.C)

        # add BOS
        l_ary = [self._bos_emb]
        p_ary = [self._pos_ids.bos_id]

        # add tables
        with torch.no_grad():
            for m in token:
                try:
                    t = self._tokenizer.encode(m.surface(), add_special_tokens=False, return_tensors="pt")
                    l_ary.append(torch.mean(self._embeddings(t), dim=1).squeeze().to(dtype=torch.bfloat16).tolist())
                    p_ary.append(self._pos_ids.get_pos_id(m.part_of_speech()))

                except RuntimeError:
                    continue

        # add EOS
        l_ary.append(self._eos_emb)
        p_ary.append(self._pos_ids.eos_id)

        return {
            "labels": torch.tensor(l_ary, dtype=torch.bfloat16).detach(),
            "inputs_poss": torch.tensor(p_ary).detach(),
            "attention_mask": torch.tensor([1 for _ in range(len(p_ary))]).detach(),
        }
