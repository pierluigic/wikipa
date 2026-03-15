# WikIPA

Code and evaluation resources for the paper:

**WikIPA: Integrating WikiPron and Lingua Libre for Multilingual IPA Transcription**. 


WikIPA is a multilingual benchmark for **speech-to-IPA (STIPA)** transcription that links **IPA transcriptions from WikiPron** with **spoken recordings from Lingua Libre**, enabling evaluation of speech-to-phonetic-transcription models across many languages.

The repository contains:

* Evaluation code used in the paper
* Model predictions for several STIPA systems
* The **test split** of the WikIPA dataset
* Scripts to reproduce evaluation heatmaps reported in the paper
* Representative samples of model predictions with error annotations


## Citation

If you use WikIPA or this repository, please cite:

```bibtex
@inproceedings{cassotti2026wikipa,
  title={WikIPA: Integrating WikiPron and Lingua Libre for Multilingual IPA Transcription},
  author={Cassotti, Pierluigi and Suchardt, Jacob Lee and De Cristofaro, Domenico},
  booktitle={Proceedings of LREC 2026},
  year={2026}
}
```

---

## Dataset

The **WikIPA dataset** integrates:

* **WikiPron**: human-curated IPA transcriptions extracted from Wiktionary
* **Lingua Libre**: crowdsourced speech recordings

The dataset contains:

* **289,694 audio–IPA pairs**
* **78 languages**
* Both **broad (phonemic)** and **narrow (phonetic)** IPA transcriptions

The dataset is available on Hugging Face:

👉 [https://huggingface.co/datasets/pierluigic/WikIPA](https://huggingface.co/datasets/pierluigic/WikIPA)

This repository includes the **test split of WikIPA** used for evaluation:

```
ground_truth.jsonl
```

Each entry contains the reference IPA transcription for the evaluation set.

---

## Repository Contents

```
.
├── evaluate.py
├── test_multipa.py
├── test_zipa.py
├── test_whipa.py
├── ground_truth.jsonl
├── multipa_predictions/
├── zipa_predictions/
├── whipa_predictions/
└── annotated_samples/
```

### Ground Truth

* **`ground_truth.jsonl`**
  Contains the **test set split of WikIPA** used in the experiments.

### Model Predictions

Pre-computed predictions for evaluated models are included:

* `multipa_predictions/`
* `zipa_predictions/`
* `whipa_predictions/`

These correspond to predictions produced by the respective STIPA systems.

### Annotated Samples of Model Predictions

* `annotated_samples`
    Includes a representative sample of each models' predictions for the English, Italian, and German broad data together with manual error anntations for qualitative evaluation.

---

## Running Predictions

The following scripts run inference with different STIPA models on the WikIPA test set:

| Script            | Model     |
| ----------------- | --------- |
| `test_multipa.py` | MultIPA   |
| `test_zipa.py`    | ZIPA      |
| `test_whipa.py`   | (Lo)WhIPA |

Each script produces predictions that can then be evaluated with the evaluation script.

Example:

```bash
python test_zipa.py
```

---

## Evaluation

The evaluation script reproduces the metrics and visualizations used in the paper.

```
evaluate.py
```

This script:

* Computes **PER (Phone Error Rate)**
* Computes **PFER (Phonetic Feature Error Rate)**
* Produces the **heatmaps presented in the paper**

Example:

```bash
python evaluate.py
```

The generated heatmaps correspond to the evaluation figures reported in the paper.







