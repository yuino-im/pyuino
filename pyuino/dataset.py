import logging
import random
from torch.utils.data import Dataset
from datasets import load_dataset


class YuinoDatasets(Dataset):
    def __init__(self, cache_dir: str, data_len=0, data_len_per=0.01):
        self._cache_dir = cache_dir
        self._dataset = load_dataset("hashimom/yukipedia", cache_dir=self._cache_dir)

        if data_len <= 0:
            data_len = int(self._dataset.num_rows["train"] * data_len_per)
        idx = list(range(self._dataset.num_rows["train"]))
        self._data_idx = random.sample(idx, data_len)
        logging.info("Data read size= %d" % data_len)

    def __len__(self):
        return len(self._data_idx)

    def __getitem__(self, idx):
        if self._dataset is None:
            self._dataset = load_dataset("hashimom/yukipedia", cache_dir=self._cache_dir)

        return {"text": self._dataset.data["train"]["text"][idx].as_py()}
