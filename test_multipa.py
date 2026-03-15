from datasets import load_dataset, Audio
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoProcessor
from transformers import pipeline

save_dir = "multipa_predictions"
os.makedirs(save_dir, exist_ok=True)

LLMs_CACHE_DIR = ''

ds = load_dataset("pierluigic/WikIPA", cache_dir=LLMs_CACHE_DIR)['test'].cast_column("audio", Audio(sampling_rate=16000))

# 2) Keep only what you need (optional but faster)
keep_cols = ["audio"]
ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

# 3) Collate: pad variable-length waveforms and keep lengths
def collate(batch):
    # batch is a list of dicts from HF datasets
    #waves = [torch.tensor(ex["audio"]["array"], dtype=torch.float32) for ex in batch]
    examples = [ex for ex in batch]
    return examples

loader = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    collate_fn=collate,
    num_workers=4,       # tune for your machine
    pin_memory=True      # if using CUDA
)


model = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
asr = pipeline("automatic-speech-recognition", model=model)


model_name = model.split('/')[1]
print(model_name)

with open(f'{save_dir}/{model_name}.txt','w+') as f:
    for j, audio in enumerate(tqdm(loader, desc="Predicting...")):
    #for example in tqdm(ds, desc="Predicting..."):
        """
        example_input = processor(
            audio=audio.get_all_samples().data,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt")#
        """
        # move all tensors in the dict to the same device as the model
        updated_audio = []
        for a in audio:
            D = {'raw':a['audio']['array'], 'sampling_rate':a['audio']['sampling_rate']}
            updated_audio.append(D)
        ipa_prediction = asr(updated_audio)
        for prediction in ipa_prediction:
            f.write(f'{prediction["text"]}\n')
