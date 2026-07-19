import argparse
import json
import torch
from transformers import TrainingArguments, GPTNeoXConfig
from pyuino import YuinoModel, YuinoTrainer, build_dictionary
from pyuino.model import YuinoConvModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# for Debug
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "YuinoLM"


def build_dict():
    build_dictionary()


def export_to_torchscript():
    model = YuinoModel.from_pretrained(model_id)
    wrapper = YuinoConvModel(model)

    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 128)
        y = wrapper(dummy_input)
        print(y.size())

    model_file_path = os.path.join(model_id, "yuino.pt")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    traced_model.save(model_file_path)
    print(f"Model saved to {model_file_path}")


def train():
    parser = argparse.ArgumentParser(description='yuinotrain')
    parser.add_argument('-d', '--data_cache_dir', default="~/hf_datasets", help="data cache path")
    parser.add_argument('-c', '--conf', default="./YuinoLM/config.json")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('--init_train', action='store_true')
    parser.add_argument('--data_len_per', type=float, default=0.05)
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
            m_config = GPTNeoXConfig.from_dict(m_config)
            model = YuinoModel(m_config)
    else:
        model = YuinoModel.from_pretrained(model_id)

    trainer = YuinoTrainer(model, training_args, args.data_cache_dir, data_len_per=args.data_len_per)
    trainer.train()
    trainer.save_model()


def main():
    train()
    build_dict()
    export_to_torchscript()


if __name__ == "__main__":
    main()
