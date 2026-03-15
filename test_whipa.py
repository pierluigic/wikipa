from datasets import load_dataset, Audio
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from whipa.code.deploy import WHIPA
import numpy as np
from transformers import AutoProcessor
import sys

par1, par2 = sys.argv[1], sys.argv[2]

save_dir = "whipa_predictions"
os.makedirs(save_dir, exist_ok=True)

LLMs_CACHE_DIR = ''

ds = load_dataset("pierluigic/WikIPA", cache_dir=LLMs_CACHE_DIR)['test'].cast_column("audio", Audio(sampling_rate=16000))

# 2) Keep only what you need (optional but faster)
keep_cols = ["audio", "orthographic_transcription"]
ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

# 3) Collate: pad variable-length waveforms and keep lengths
def collate(batch):
    # batch is a list of dicts from HF datasets
    #waves = [torch.tensor(ex["audio"]["array"], dtype=torch.float32) for ex in batch]
    examples = [{"audio": ex["audio"]} for ex in batch]
    texts = [ex.get("orthographic_transcription", "") for ex in batch]
    return examples, texts

batch_size = 64

if par1 == 'large':
    batch_size = 32

loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
    num_workers=4,       # tune for your machine
    pin_memory=True      # if using CUDA
)


def get_whipa_model(model_path):
    model = WHIPA(model_path=model_path, cache_dir=LLMs_CACHE_DIR)  
    return model


def get_lowhipamodel(model_path):
    model = WHIPA(model_path=model_path, lora=True, cache_dir=LLMs_CACHE_DIR)
    return model


models = {
    'WHIPA': ['jshrdt/whipa-base-cv', 'jshrdt/whipa-large-cv'],
    'LOWHIPA': ['jshrdt/lowhipa-base-cv', 'jshrdt/lowhipa-large-cv', 'jshrdt/lowhipa-base-asc', 'jshrdt/lowhipa-large-asc', 'jshrdt/lowhipa-base-comb', 'jshrdt/lowhipa-large-comb']
}

model_initializers = {'WHIPA':get_whipa_model, 'LOWHIPA':get_lowhipamodel}


for approach in models:

    for model in models[approach]:

        model_name = model.split('/')[1]

        if par1 in model and model_name.startswith(par2):
            print(model_name)
            
            model = model_initializers[approach](model)
            processor = model.processor

            with open(f'{save_dir}/{model_name}.txt','w+') as f:
                for j, (audio, texts) in enumerate(tqdm(loader, desc="Predicting...")):
                #for example in tqdm(ds, desc="Predicting..."):
                    """
                    example_input = processor(
                        audio=audio.get_all_samples().data,
                        sampling_rate=16000,
                        padding=True,
                        return_tensors="pt")#
                    """
                    # move all tensors in the dict to the same device as the model
                    ipa_prediction = model.transcribe_ipa_batch(audio, fallback=False, verbose=False)
                    for prediction in ipa_prediction:
                        f.write(f'{prediction}\n')
