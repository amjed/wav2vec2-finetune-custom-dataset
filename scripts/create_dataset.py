import argparse
import math
import pickle
import re

import inflect
import librosa
import pandas as pd
import srt
from datasets import Dataset

remove_regex = r'\d|\+|:|\[|\]'

output_pick = '../data/ict_finetune_dataset.pkl'
p = inflect.engine()


def get_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    return y


def get_sample(s, ts, te, sr=16000):
    step = (1 / sr)
    pad_left = int(0.05 / step)
    pad_right = int(0.1 / step)
    idx_start = math.floor(ts / step)
    idx_end = math.ceil(te / step)
    if idx_start - pad_left > 0:
        idx_start = idx_start - pad_left
    if idx_end + pad_right < len(s):
        idx_end = idx_end + pad_right
    return s[idx_start:idx_end]


def make_entry(audio, text, sr=16000):
    return {
        "audio": audio,
        "sampling_rate": sr,
        "text": text
    }


def generate_entries(subs, audio_samples):
    entries = []
    for sub in subs:
        text = sub.content
        if re.search(r"\[|\]", text):
            print(f'ignoring {text}')
            continue
        for m in re.findall(r'\d+', text):
            text = text.replace(m, p.number_to_words(int(m))).replace('-', ' ')
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        samples = get_sample(audio_samples, start, end)
        entries.append(make_entry(samples, text))
    return entries


def make_df_entry(data):
    cols = ["audio", "sampling_rate", "text"]
    df = pd.DataFrame(data, columns=cols)
    return df


def run_cli(dataset_name):
    srt_files = [
        '../data/1.srt',
        '../data/2.srt',
    ]

    audio_files = [
        '../data/1.wav',
        '../data/2.wav',
    ]

    entries = []

    for srt_file, audio_file in zip(srt_files, audio_files):
        with open(srt_file, 'r') as f:
            subs = list(srt.parse(f.read()))
        s = get_audio(audio_file)
        entries += generate_entries(subs, s)
    print(len(entries))
    dataset = make_df_entry(entries)
    ict_dataset = Dataset.from_pandas(dataset)
    with open(dataset_name, 'wb') as outfile:
        pickle.dump(ict_dataset, outfile)


def run():
    args = parser()
    run_cli(args.dataset)


def parser():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--dataset', type=str, help='pickled raw dataset to generate', nargs='?',
                                 required=True)
    return argument_parser.parse_args()


if __name__ == "__main__":
    run()
