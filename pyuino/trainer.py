import os
import torch
from typing import Tuple
from collections import OrderedDict
from torch.utils.data.dataset import Subset
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from .model import YuinoModel
from .dataset import YuinoDatasets
from .dictionary import YuinoDicPosId

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class YuinoCollator:
    def __init__(self, teacher_model: AutoModel, teacher_tokenizer: AutoTokenizer, cache_size: int = 2048):
        teacher_model.eval()
        self._embeddings = teacher_model.get_input_embeddings()
        self._tokenizer = teacher_tokenizer
        self._pos_ids = YuinoDicPosId()
        self._cache_size = cache_size
        self._surface_cache = OrderedDict()

        with torch.no_grad():
            bos_ids = self._tokenizer.encode("[CLS]", add_special_tokens=False, return_tensors="pt")
            eos_ids = self._tokenizer.encode("[SEP]", add_special_tokens=False, return_tensors="pt")
            self._bos_emb = self._embeddings(bos_ids).mean(dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
            self._eos_emb = self._embeddings(eos_ids).mean(dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
        del bos_ids, eos_ids

    def _get_surface_embedding(self, surface: str) -> torch.Tensor:
        cached = self._surface_cache.get(surface)
        if cached is not None:
            self._surface_cache.move_to_end(surface)
            return cached

        with torch.no_grad():
            token_ids = self._tokenizer.encode(surface, add_special_tokens=False, return_tensors="pt")
            emb = self._embeddings(token_ids).mean(dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()

        self._surface_cache[surface] = emb
        self._surface_cache.move_to_end(surface)

        if len(self._surface_cache) > self._cache_size:
            self._surface_cache.popitem(last=False)

        del token_ids
        return emb

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

    def __call__(self, batch):
        labels = []
        inputs_poss = []
        attention_mask = []

        for item in batch:
            pairs = self._parse_text(item["text"])

            sample_labels = [self._bos_emb]
            sample_pos_ids = [self._pos_ids.bos_id]

            for surface, pos_id in pairs:
                emb = self._get_surface_embedding(surface)
                sample_labels.append(emb)
                sample_pos_ids.append(pos_id)

            sample_labels.append(self._eos_emb)
            sample_pos_ids.append(self._pos_ids.eos_id)

            labels.append(torch.stack(sample_labels, dim=0))
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
            teacher_model: AutoModel,
            teacher_tokenizer: AutoTokenizer,
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
        data_collator = YuinoCollator(teacher_model, teacher_tokenizer)

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
