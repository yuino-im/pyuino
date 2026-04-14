import logging
from torch.utils.data import Dataset
from datasets import load_dataset


class YuinoDatasets(Dataset):
    def __init__(self, cache_dir: str, dataset_len=402924252, data_len_per=0.01):
        self._dataset = None
        self._cache_dir = cache_dir

        self._data_len = int(dataset_len * data_len_per)
        logging.info("Data read size= %d" % self._data_len)

    def __len__(self):
        return self._data_len

    def __getitem__(self, idx):
        if self._dataset is None:
            self._dataset = load_dataset("hashimom/yukipedia", cache_dir=self._cache_dir)

        return {"text": self._dataset.data["train"]["text"][idx].as_py().strip()}
