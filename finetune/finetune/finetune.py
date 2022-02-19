import argparse
import os
import pickle

from datasets import load_metric, set_caching_enabled
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, \
    Wav2Vec2FeatureExtractor

from finetune.DataCollatorCTCWithPadding import DataCollatorCTCWithPadding

set_caching_enabled(False)

import numpy as np

# Set environment variables
os.environ['WANDB_DISABLED '] = 'True'


def run_train(output_model_name, base_xlsr_model, ds, vocab_json_file):
    tokenizer = Wav2Vec2CTCTokenizer(vocab_json_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(f'./{output_model_name}')

    def prepare_dataset(batch):
        batch["input_values"] = processor(np.asarray(batch['audio']), sampling_rate=16000).input_values[0]

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

    prepared_ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=8)

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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=3e-4,
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

    return ict_ds


def main(args: argparse.Namespace):
    output_model_name = args.tuned
    base_model_name = args.base
    dataset_pickle = args.dataset
    vocab_file = args.vocab
    ds = get_train_test_sets(dataset_pickle)
    run_train(output_model_name, base_model_name, ds, vocab_file)


def parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--dataset', type=str, help='pickled raw dataset to be split in script', nargs='?',
                                 required=True)
    argument_parser.add_argument('--vocab', type=str, help='vocab json file', nargs='?', required=True)
    argument_parser.add_argument('--base', type=str, help='base xslr model name', nargs='?', required=True)
    argument_parser.add_argument('--tuned', type=str, help='fine-tuned model name', nargs='?', required=True)
    return argument_parser.parse_args()


def run():
    args = parser()
    main(args)


if __name__ == "__main__":
    run()
