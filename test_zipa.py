from datasets import load_dataset, Audio
import torchaudio
import torch
from zipa_feature_extractor import ZIPA_FeatureExtractor
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from natsort import natsorted 

"""
save_dir = "zipa_predictions"
os.makedirs(save_dir, exist_ok=True)

LLMs_CACHE_DIR = ''

ds = load_dataset("pierluigic/WikIPA", cache_dir=LLMs_CACHE_DIR)['test'].cast_column("audio", Audio(sampling_rate=16000))



# 2) Keep only what you need (optional but faster)
keep_cols = ["audio"]
ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

# 3) Collate: pad variable-length waveforms and keep lengths
def collate(batch):
    # batch is a list of dicts from HF datasets
    waves = [torch.tensor(ex["audio"]["array"], dtype=torch.float32) for ex in batch]
    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.int32)
    waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)  # (B, Tmax)
    return waves, lengths

loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    collate_fn=collate,
    num_workers=4,       # tune for your machine
    pin_memory=True      # if using CUDA
)


feature_extractor = ZIPA_FeatureExtractor()

features_save_dir = f"{save_dir}/features"
os.makedirs(features_save_dir, exist_ok=True)

for j, (waves, lengths) in enumerate(tqdm(loader, desc="Extracting features...")):
    waves = waves
    lengths = lengths
    feats, feats_lens = feature_extractor.inference(waves)   

    torch.save({
        "feats": feats.cpu(),
        "feats_lens": feats_lens.cpu(),
    }, os.path.join(features_save_dir, f"batch_{j:05d}.pt"))
"""

features_save_dir = "zipa_predictions/features"
bpe_model_path = "models/unigram_127.model"

def get_ctc_model(model_path): 
    from zipa_ctc_inference import initialize_model
    
    model = initialize_model(model_path, bpe_model_path) 
    
    return model

def get_transducer_model(model_path): 
    from zipa_transducer_inference import initialize_model
    
    model = initialize_model(model_path, bpe_model_path) 
    
    return model

models = {
    'CTC': ['zipa_small_crctc_extended_0.5_scale_700000_avg10.pth', 'zipa_large_crctc_0.5_scale_800000_avg10.pth'],
    #'TRANSDUCER': ['zipa_small_noncausal_500000_avg10.pth','zipa_large_noncausal_500000_avg10.pth']
}

model_initializers = {'CTC':get_ctc_model, 'TRANSDUCER':get_transducer_model}


for approach in models:

    for model in models[approach]:

        model_path = f"models/{model}"

        model_name = '.'.join(model.split('.')[:-1])
        print(model_name)

        model = model_initializers[approach](model_path)

        batch_files = natsorted([f for f in os.listdir(features_save_dir) if f.endswith(".pt")])

        with open(f'zipa_predictions/{model_name}.txt','w+') as f:
            for batch_file in tqdm(batch_files, 'Processing batches'):
                batch_path = os.path.join(features_save_dir, batch_file)
                data = torch.load(batch_path, map_location="cpu")

                feats = data["feats"]         # (B, T, F)
                feats_lens = data["feats_lens"]  # (B,)

                output = model.inference(feats, feats_lens)
            
                for example in output:
                    if len(example)>0:
                        f.write(f'{example[0]}\n')
                    else:
                        f.write(f'\n')