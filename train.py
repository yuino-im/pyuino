import argparse
import json
from transformers import TrainingArguments, AutoModel, AutoTokenizer, Qwen3Config
from pyuino import YuinoModel, YuinoTrainer, build_dictionary

import os
base_model_path = "line-corporation/line-distilbert-base-japanese"

model_id = "YuinoLM"


def build_dict():
    model = AutoModel.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    build_dictionary(model, tokenizer)


def train():
    parser = argparse.ArgumentParser(description='yuinotrain')
    parser.add_argument('-d', '--data_cache_dir', default="~/hf_datasets", help="data cache path")
    parser.add_argument('-c', '--conf', default="./YuinoLM/config.json")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('--init_train', action='store_true')
    parser.add_argument('--data_len_per', type=float, default=0.1)
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir="YuinoLM",
        eval_strategy="epoch",
        learning_rate=1e-4,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        num_train_epochs=args.epoch,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=16,
        dataloader_num_workers=8,
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

    tcr_model = AutoModel.from_pretrained(base_model_path)
    tcr_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model.config.use_cache = False
    trainer = YuinoTrainer(model, tcr_model, tcr_tokenizer, training_args, args.data_cache_dir, data_len_per=args.data_len_per)
    trainer.train()
    trainer.save_model()


def main():
    train()
    #convert_onnx()
    build_dict()


if __name__ == "__main__":
    main()
