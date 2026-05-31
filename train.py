import argparse
import json
from transformers import TrainingArguments, Qwen3Config
from pyuino import YuinoModel, YuinoTrainer, build_dictionary

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# for Debug
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "YuinoLM"


def build_dict():
    build_dictionary()


def train():
    parser = argparse.ArgumentParser(description='yuinotrain')
    parser.add_argument('-d', '--data_cache_dir', default="~/hf_datasets", help="data cache path")
    parser.add_argument('-c', '--conf', default="./YuinoLM/config.json")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('--init_train', action='store_true')
    parser.add_argument('--data_len_per', type=float, default=0.02)
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir="YuinoLM",
        eval_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        num_train_epochs=args.epoch,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=16,
        dataloader_num_workers=4,
        bf16=True,
        remove_unused_columns=False,
    )

    if args.init_train:
        with open(args.conf, 'r') as f:
            m_config = json.load(f)
            m_config = Qwen3Config.from_dict(m_config)
            model = YuinoModel(m_config)
    else:
        model = YuinoModel.from_pretrained(model_id)

    trainer = YuinoTrainer(model, training_args, args.data_cache_dir, data_len_per=args.data_len_per)
    trainer.train()
    trainer.save_model()


def main():
    train()
    build_dict()


if __name__ == "__main__":
    main()
