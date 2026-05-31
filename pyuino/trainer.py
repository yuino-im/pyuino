import os
import torch
import fasttext
from typing import Tuple
from torch.utils.data.dataset import Subset
from transformers import Trainer, TrainingArguments
from torch import nn
from .model import YuinoModel
from .dataset import YuinoDatasets
from .dictionary import YuinoDicPosId

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class YuinoCollator:
    def __init__(self, fs_model_path="./model.bin"):
        self._pos_ids = YuinoDicPosId()
        self._ft_model = fasttext.load_model(fs_model_path)
        self._sigmoid = nn.Sigmoid()
        self._bos_emb = self.get_vector("__BOS")
        self._eos_emb = self.get_vector("__EOS")

    @staticmethod
    def _parse_text(text: str) -> list[tuple[str, int]]:
        pairs = []
        for m in text.split(" "):
            parts = m.split("/")
            if len(parts) != 2:
                continue

            try:
                surface = parts[0]
                pos_id = int(parts[1])
            except ValueError:
                continue

            pairs.append((surface, pos_id))

        return pairs

    def get_vector(self, input_text: str):
        x = self._sigmoid(torch.tensor(self._ft_model[input_text], dtype=torch.bfloat16))
        return torch.where((x > 0.5), 1., 0.).to(x.dtype)

    def __call__(self, batch):
        labels = []
        inputs_poss = []
        attention_mask = []

        for item in batch:
            pairs = self._parse_text(item["text"])

            sample_labels = [self._bos_emb]
            sample_pos_ids = [self._pos_ids.bos_id]

            for surface, pos_id in pairs:
                sample_labels.append(self.get_vector(surface))
                sample_pos_ids.append(pos_id)

            sample_labels.append(self._eos_emb)
            sample_pos_ids.append(self._pos_ids.eos_id)

            labels.append(torch.stack(sample_labels))
            inputs_poss.append(torch.tensor(sample_pos_ids, dtype=torch.long))
            attention_mask.append(torch.ones(len(sample_pos_ids), dtype=torch.long))
            del sample_labels, sample_pos_ids

        result = {
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
            "inputs_poss": torch.nn.utils.rnn.pad_sequence(inputs_poss, batch_first=True),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
        }
        del labels, inputs_poss, attention_mask
        return result


class YuinoTrainer(Trainer):
    def __init__(
            self,
            model: YuinoModel,
            training_args: TrainingArguments,
            data_cache_dir: str,
            data_len_per: float = 0.01,
            valid_len_per: float = 0.1,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        all_dataset = YuinoDatasets(cache_dir=data_cache_dir, data_len_per=data_len_per)
        n_samples = len(all_dataset)
        train_len = int(n_samples - (n_samples * valid_len_per))
        train_dataset = Subset(all_dataset, list(range(0, train_len)))
        valid_dataset = Subset(all_dataset, list(range(train_len, n_samples)))
        data_collator = YuinoCollator()

        super(YuinoTrainer, self).__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            callbacks=None,
            optimizers=optimizers,
            processing_class=None,
            compute_metrics=None,
        )
