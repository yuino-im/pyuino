import argparse
import json
from transformers import TrainingArguments, AutoModel, AutoTokenizer, Qwen3Config
from pyuino import YuinoModel, YuinoDictionary, YuinoTrainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
base_model_path = "line-corporation/line-distilbert-base-japanese"

model_id = "YuinoLM"

def build_dictionary():
    model = AutoModel.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    YuinoDictionary.build(model, tokenizer)

def train():
    parser = argparse.ArgumentParser(description='yuinotrain')
    parser.add_argument('-d', '--data_cache_dir', default="./dataset", help="data cache path")
    parser.add_argument('-c', '--conf', default="./YuinoLM/config.json")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('--init_train', action='store_true')
    parser.add_argument('--data_len_per', type=float, default=0.01)
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir="YuinoLM",
        eval_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        num_train_epochs=args.epoch,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=8,
    )

    if args.init_train:
        with open(args.conf, 'r') as f:
            m_config = json.load(f)
            m_config = Qwen3Config.from_dict(m_config)
            model = YuinoModel(m_config)
    else:
        model = YuinoModel.from_pretrained("YuinoLM")

    tcr_model = AutoModel.from_pretrained(base_model_path)
    tcr_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    trainer = YuinoTrainer(model, tcr_model, tcr_tokenizer, training_args, args.data_cache_dir, data_len_per=args.data_len_per)
    trainer.train()
    trainer.save_model()


def main():
    train()
    #convert_onnx()
    build_dictionary()


if __name__ == "__main__":
    main()
