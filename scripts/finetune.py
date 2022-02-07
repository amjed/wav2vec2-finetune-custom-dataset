import argparse
import json
import os
import pickle
import re

from datasets import load_metric, set_caching_enabled
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, \
    Wav2Vec2FeatureExtractor

from scripts.DataCollatorCTCWithPadding import DataCollatorCTCWithPadding

set_caching_enabled(False)

import numpy as np

# Set environment variables
os.environ['WANDB_DISABLED '] = 'True'

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\+\d]'


def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def run_train(output_model_name, base_xlsr_model, ds):
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(f'./{output_model_name}')

    def prepare_dataset(batch):
        batch["input_values"] = processor(batch['audio'], sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # prepared_ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=4)
    prepared_ds = ds.map(prepare_dataset, remove_columns=ds.column_names['train'], num_proc=4)
    max_input_length_in_sec = 4.0
    prepared_ds = prepared_ds.filter(
        lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
        input_columns=["input_length"])

    # train = train.map(prepare_dataset, remove_columns=train.column_names)
    # test = test.map(prepare_dataset, remove_columns=test.column_names)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        base_xlsr_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=output_model_name,
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds['train'],
        eval_dataset=prepared_ds['test'],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    model.save_pretrained(output_model_name)
    processor.save_pretrained(output_model_name)


def get_train_test_sets(dataset_pickle):
    infile = open(dataset_pickle, 'rb')
    raw_ds = pickle.load(infile)

    ict_ds = raw_ds.train_test_split(0.2)

    ict_ds = ict_ds.map(remove_special_characters)

    vocabs = ict_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                        remove_columns=ict_ds.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_list.sort()

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return ict_ds


def main(args: argparse.Namespace):
    output_model_name = args.tuned
    base_model_name = args.base
    dataset_pickle = args.dataset
    ds = get_train_test_sets(dataset_pickle)
    run_train(output_model_name, base_model_name, ds)


def parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--dataset', type=str, help='pickled raw dataset to be split in script', nargs='?',
                                 required=True)
    argument_parser.add_argument('--base', type=str, help='base xslr model name', nargs='?', required=True)
    argument_parser.add_argument('--tuned', type=str, help='fine-tuned model name', nargs='?', required=True)
    return argument_parser.parse_args()


def run():
    args = parser()
    main(args)


if __name__ == "__main__":
    run()
