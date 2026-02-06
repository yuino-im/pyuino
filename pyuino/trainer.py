import os
import torch
from typing import Tuple
from torch.utils.data.dataset import Subset
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from .model import YuinoModel
from .dataset import YuinoDatasets

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def collate_fn(batch):
    inputs_poss = []
    labels = []
    attention_mask = []
    for d in batch:
        labels.append(d["labels"])
        inputs_poss.append(d["inputs_poss"])
        attention_mask.append(d["attention_mask"])

    return {
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
        "inputs_poss": torch.nn.utils.rnn.pad_sequence(inputs_poss, batch_first=True),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    }


class YuinoTrainer(Trainer):
    def __init__(
            self,
            model: YuinoModel,
            teacher_model: AutoModel,
            teacher_tokenizer: AutoTokenizer,
            training_args: TrainingArguments,
            data_cache_dir: str,
            data_len_per: float = 0.01,
            valid_len_per: float = 0.1,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        all_dataset = YuinoDatasets(teacher_model, teacher_tokenizer, cache_dir=data_cache_dir, data_len_per=data_len_per)
        n_samples = len(all_dataset)
        train_len = int(n_samples - (n_samples * valid_len_per))
        train_dataset = Subset(all_dataset, list(range(0, train_len)))
        valid_dataset = Subset(all_dataset, list(range(train_len, n_samples)))

        super(YuinoTrainer, self).__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collate_fn,
            callbacks=None,
            optimizers=optimizers,
            processing_class=None,
            compute_metrics=None,
        )
