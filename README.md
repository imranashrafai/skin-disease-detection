# DermAI — Multimodal Skin Disease Detection

**Two AI models. One diagnosis.**

| Layer | Model | Input | Output |
|---|---|---|---|
| 1 | MobileNetV2 CNN (TensorFlow) | Skin image | Disease probabilities |
| 2 | BioBERT NLP (PyTorch) | Symptom text | Disease probabilities |
| Fusion | Soft-vote (65% CNN + 35% BioBERT) | Both | Final prediction |

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download and prepare image dataset
```bash
# Set up Kaggle API key first:
# 1. Go to https://www.kaggle.com/settings → API → Create New Token
# 2. Place kaggle.json in ~/.kaggle/kaggle.json
# 3. chmod 600 ~/.kaggle/kaggle.json

python setup_datasets.py
```
This downloads the **31-class skin disease image dataset** and the **HuggingFace clinical CSVs**.

### Step 3 — Train the CNN image model
```bash
python train_model.py
```
Outputs to `model/`:
- `skin_disease_model.h5` — trained CNN weights
- `class_names.json` — 31 class names
- `disease_info.json` — symptoms, treatments, severity per disease
- `ood_threshold.json` — out-of-distribution cutoff

### Step 4 — Download the text dataset
Download from Kaggle:
> https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Place the CSV in the project root. The script auto-detects these filenames:
- `disease_symptom_description.csv`
- `symptom_Description.csv`
- `dataset.csv`

### Step 5 — Train the BioBERT text model
```bash
python train_text_model.py
```
Uses `model/class_names.json` (from Step 3) to align labels with the CNN.

Outputs to `model/`:
- `text_model/` — fine-tuned BioBERT directory
- `text_label_map.json` — {disease: index} mapping

Custom CSV path or hyperparameters:
```bash
python train_text_model.py --csv my_symptoms.csv --epochs 5 --batch 16 --lr 2e-5
```

### Step 6 — Run the web app
```bash
python app.py
```
Open http://localhost:5000

---

## How the fusion works

```
Image uploaded  →  CNN  →  probs_image[31]
Text entered    →  BioBERT  →  probs_text[31]

final = 0.65 × probs_image + 0.35 × probs_text
predicted_class = argmax(final)
```

If **no text** is provided, only the CNN runs (`weight = 1.0 / 0.0`).
The OOD (out-of-distribution) check is applied only to the CNN's raw
confidence, so non-skin images are caught before text is considered.

---

## Project structure

```
skin-disease-detector/
│
├── app.py                  ← Flask backend (dual-model inference + fusion)
├── train_model.py          ← CNN training (MobileNetV2 + HF enrichment)
├── train_text_model.py     ← BioBERT training (symptom text classification)
├── setup_datasets.py       ← Dataset download automation
├── requirements.txt
│
├── templates/
│   └── index.html          ← Frontend (upload + symptom input + fusion UI)
│
├── static/uploads/         ← Temp image storage (auto-deleted after 60s)
│
├── dataset/
│   ├── train/              ← 31 class folders for CNN training
│   └── test/               ← 31 class folders for CNN evaluation
│
└── model/
    ├── skin_disease_model.h5
    ├── class_names.json
    ├── disease_info.json
    ├── ood_threshold.json
    ├── text_model/             ← BioBERT fine-tuned weights
    ├── text_label_map.json
    └── final_report.json
```

---

## Tech stack

| Component | Technology |
|---|---|
| Web framework | Flask + Flask-CORS |
| CNN architecture | MobileNetV2 (ImageNet pre-trained) |
| CNN framework | TensorFlow / Keras |
| NLP model | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) |
| NLP framework | PyTorch + HuggingFace Transformers |
| Image dataset | Kaggle 31-class skin disease dataset |
| Text dataset | Disease Symptom Description Dataset (Kaggle) |
| Clinical enrichment | ElectricSheepAfrica HuggingFace CSVs |
| Frontend | Vanilla JS, CSS custom properties |

---

Developed By: Imran Ashraf  
Generative AI R&D Engineer at DeepCogSol
