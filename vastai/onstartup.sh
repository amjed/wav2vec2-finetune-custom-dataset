#!/bin/sh

echo "[+] starting..."
echo "[+] starting finetuning"
tuned_name="xlsr-tuned-$(date '+%Y-%m-%d-%H%M')"
fine_tune_custom --dataset /root/finetune_dataset.pkl --vocab /root/vocab.json --base "jonatasgrosman/wav2vec2-large-xlsr-53-english" --tuned $tuned_name
