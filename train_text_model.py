"""
train_text_model.py — DermAI Text Model Training
==================================================
Fine-tunes BioBERT on the Disease Symptom Description Dataset from Kaggle.
Maps free-text symptom descriptions → one of the 31 skin disease classes.

Dataset:
  Kaggle: "Disease Symptom Description Dataset"
  URL: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
  OR any CSV with columns: [disease, symptom_1 ... symptom_17] or [disease, symptoms_text]

HOW IT WORKS:
  1. Load disease-symptom CSV
  2. Filter to only classes that exist in class_names.json (from CNN training)
  3. Convert symptom columns → single text string per row
  4. Fine-tune BioBERT (dmis-lab/biobert-base-cased-v1.2) as a classifier
  5. Save: model/text_model/  and  model/text_label_map.json

OUTPUTS:
  model/text_model/           ← HuggingFace model directory (pytorch_model.bin etc.)
  model/text_label_map.json   ← {disease_name: class_index} matching CNN order
  model/text_training_report.json

USAGE:
  python train_text_model.py
  python train_text_model.py --csv path/to/your_symptoms.csv
  python train_text_model.py --epochs 5 --batch 16

Requirements:
  pip install transformers torch scikit-learn pandas datasets accelerate
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BIOBERT_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
MODEL_DIR       = "model"
TEXT_MODEL_DIR  = os.path.join(MODEL_DIR, "text_model")
CLASS_NAMES_PATH= os.path.join(MODEL_DIR, "class_names.json")
LABEL_MAP_PATH  = os.path.join(MODEL_DIR, "text_label_map.json")
REPORT_PATH     = os.path.join(MODEL_DIR, "text_training_report.json")
MAX_LEN         = 256
SEED            = 42

# ── Disease name normalisation map ───────────────────────────────────────────
# Maps symptom-dataset disease names → CNN class names (case-insensitive keys)
DISEASE_NAME_MAP = {
    "acne": "Acne",
    "fungal infection": "Ringworm",
    "allergy": "Urticaria",
    "gerd": None,                        # not a skin disease — skip
    "chronic cholestasis": None,
    "drug reaction": "Drug Eruption",
    "peptic ulcer disease": None,
    "aids": None,
    "diabetes": None,
    "gastroenteritis": None,
    "bronchial asthma": None,
    "hypertension": None,
    "migraine": None,
    "cervical spondylosis": None,
    "paralysis (brain hemorrhage)": None,
    "jaundice": None,
    "malaria": None,
    "chicken pox": "Chickenpox",
    "chickenpox": "Chickenpox",
    "dengue": None,
    "typhoid": None,
    "hepatitis a": None,
    "hepatitis b": None,
    "hepatitis c": None,
    "hepatitis d": None,
    "hepatitis e": None,
    "alcoholic hepatitis": None,
    "tuberculosis": None,
    "common cold": None,
    "pneumonia": None,
    "dimorphic hemorrhoids (piles)": None,
    "heart attack": None,
    "varicose veins": None,
    "hypothyroidism": None,
    "hyperthyroidism": None,
    "hypoglycemia": None,
    "osteoarthritis": None,
    "arthritis": None,
    "vertigo": None,
    "acne": "Acne",
    "psoriasis": "Psoriasis",
    "impetigo": "Impetigo",
    "eczema": "Eczema",
    "melanoma": "Melanoma",
    "ringworm": "Ringworm",
    "urticaria": "Urticaria",
    "vitiligo": "Vitiligo",
    "scabies": "Scabies",
    "warts": "Warts",
    "shingles": "Shingles",
    "herpes": "Herpes Simplex",
    "herpes simplex": "Herpes Simplex",
    "monkeypox": "Monkeypox",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "squamous cell carcinoma": "Squamous Cell Carcinoma",
    "skin cancer": "Skin Cancer",
    "tinea corporis": "Tinea Corporis",
    "cowpox": "Cowpox",
}

# ── Symptom keyword augmentation ─────────────────────────────────────────────
# For diseases that might not appear in the symptom CSV, we generate
# synthetic training examples from the static disease info in train_model.py.
SYNTHETIC_SYMPTOMS = {
    "Acne": [
        "itchy pimples oily skin blackheads whiteheads red bumps face",
        "painful cysts oily skin redness scarring acne vulgaris",
        "blackheads whiteheads pustules on face oily skin comedones",
    ],
    "Eczema": [
        "dry itchy inflamed skin red patches cracked skin weeping blisters",
        "intense itching dry skin redness thickened skin atopic dermatitis",
        "eczema flare inflamed itchy rash dry patches scaling",
    ],
    "Psoriasis": [
        "thick silvery scales dry cracked skin burning itching stiff joints",
        "scaly red patches itching psoriasis plaques nail pitting",
        "silvery scales inflamed skin psoriasis plaque type",
    ],
    "Melanoma": [
        "asymmetric mole irregular border multiple colours growing lesion",
        "dark irregular mole changing size shape colour melanoma",
        "skin cancer mole asymmetry border irregularity diameter evolving",
    ],
    "Basal Cell Carcinoma": [
        "pearly waxy bump flat lesion bleeding sore pink growth skin cancer",
        "non-healing sore translucent bump skin cancer basal cell",
        "pearly nodule telangiectasia rolled border skin lesion",
    ],
    "Squamous Cell Carcinoma": [
        "firm red nodule crusty surface wart-like growth ulcer squamous",
        "scaly lesion non-healing ulcer rough skin squamous cell carcinoma",
        "keratinising lesion crusty sore skin cancer squamous",
    ],
    "Melanoma": [
        "asymmetric mole irregular border multiple colours diameter growing",
        "dark mole changing size colour melanoma urgent",
    ],
    "Ringworm": [
        "ring-shaped red patch scaly itching ringworm tinea fungal infection",
        "circular itchy rash with clear centre fungal ringworm",
        "red ring itching scaly border central clearing tinea",
    ],
    "Chickenpox": [
        "itchy blisters rash fever chickenpox varicella vesicles crusting",
        "red spots fluid filled blisters fever chickenpox all over body",
    ],
    "Impetigo": [
        "red sores honey crust blisters impetigo contagious bacterial",
        "crusty yellow sores around nose mouth impetigo",
    ],
    "Scabies": [
        "intense night itching pimple rash burrow tracks mite infestation",
        "severe nocturnal itching burrows between fingers scabies mite",
    ],
    "Shingles": [
        "burning pain stripe of blisters one side body shingles herpes zoster",
        "painful rash along nerve shingles post herpetic neuralgia",
    ],
    "Herpes Simplex": [
        "cold sore blister tingling burning herpes simplex HSV",
        "fluid filled blisters mouth genitals herpes recurrent outbreak",
    ],
    "Vitiligo": [
        "white patches loss of skin colour vitiligo depigmentation",
        "pale white patches symmetrical vitiligo melanocyte loss",
    ],
    "Urticaria": [
        "raised red welts intense itching hives urticaria allergic",
        "swollen itchy wheals appear suddenly urticaria hives",
    ],
    "Warts": [
        "rough grainy growth flesh coloured bump HPV wart",
        "black pinpoints rough skin wart viral HPV",
    ],
    "Monkeypox": [
        "fever swollen lymph nodes pustular rash spreading monkeypox",
        "rash fever lymphadenopathy monkeypox viral pox lesions",
    ],
    "Tinea Corporis": [
        "ring shaped patch red scaly border itching tinea corporis",
        "circular scaly fungal patch trunk limbs tinea",
    ],
    "Drug Eruption": [
        "rash after medication drug reaction skin eruption allergic",
        "maculopapular rash drug induced hypersensitivity",
    ],
    "Cowpox": [
        "red lesion fluid filled pustule fever lymph node swelling cowpox",
        "localised pox lesion animal contact fever cowpox",
    ],
    "Skin Cancer": [
        "changing mole new growth non healing sore rough patch skin cancer",
        "bleeding lesion irregular border skin cancer multiple types",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train BioBERT text classifier for DermAI")
    p.add_argument("--csv",     default="", help="Path to disease-symptom CSV file")
    p.add_argument("--epochs",  type=int, default=4, help="Training epochs (default: 4)")
    p.add_argument("--batch",   type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--lr",      type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    p.add_argument("--max_len", type=int, default=MAX_LEN, help="Max token length (default: 256)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Load CNN class names — text model MUST output same classes in same order
# ══════════════════════════════════════════════════════════════════════════════
def load_cnn_classes() -> list:
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            classes = json.load(f)
        log.info("CNN class names loaded: %d classes", len(classes))
        return classes
    else:
        log.warning(
            "class_names.json not found at %s. "
            "Run train_model.py first, OR the text model will use its own label set.",
            CLASS_NAMES_PATH,
        )
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Load and parse symptom CSV
# ══════════════════════════════════════════════════════════════════════════════
def load_symptom_csv(csv_path: str) -> pd.DataFrame:
    """
    Handles multiple CSV layouts:
      Layout A: columns = [Disease, Symptom_1, Symptom_2, ..., Symptom_17]
      Layout B: columns = [disease, symptoms]  (single text column)
      Layout C: columns = [label, text]
    Returns DataFrame with columns: [disease, text]
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info("CSV shape: %s  columns: %s", df.shape, list(df.columns))

    # Layout B / C — already has a single text column
    text_col = next((c for c in df.columns if c in ("symptoms", "text", "description", "symptom_text")), None)
    disease_col = next((c for c in df.columns if c in ("disease", "label", "diagnosis", "class")), None)

    if disease_col and text_col:
        df = df[[disease_col, text_col]].rename(columns={disease_col: "disease", text_col: "text"})
        df = df.dropna()
        df["text"] = df["text"].astype(str).str.strip()
        log.info("Layout B/C detected. Rows after dropna: %d", len(df))
        return df

    # Layout A — multiple symptom columns
    symptom_cols = [c for c in df.columns if c.startswith("symptom")]
    if disease_col and symptom_cols:
        def row_to_text(row):
            parts = [str(row[c]).strip().replace("_", " ")
                     for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip().lower() != "nan"]
            return " ".join(parts)
        df["text"] = df.apply(row_to_text, axis=1)
        df = df[[disease_col, "text"]].rename(columns={disease_col: "disease"})
        df = df[df["text"].str.strip() != ""]
        log.info("Layout A detected (%d symptom cols). Rows: %d", len(symptom_cols), len(df))
        return df

    raise ValueError(
        f"Cannot parse CSV. Expected columns like [Disease, Symptom_1...] or [disease, symptoms].\n"
        f"Found: {list(df.columns)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Filter to skin-disease classes and map names to CNN classes
# ══════════════════════════════════════════════════════════════════════════════
def filter_and_map(df: pd.DataFrame, cnn_classes: list) -> pd.DataFrame:
    cnn_lower = {c.lower(): c for c in cnn_classes}

    def map_disease(raw: str) -> str | None:
        raw_l = raw.lower().strip()
        # Direct match against CNN classes
        if raw_l in cnn_lower:
            return cnn_lower[raw_l]
        # Check our manual map
        mapped = DISEASE_NAME_MAP.get(raw_l)
        if mapped is None:
            return None
        # Mapped value must also exist in CNN classes
        if mapped.lower() in cnn_lower:
            return cnn_lower[mapped.lower()]
        return None

    df = df.copy()
    df["cnn_class"] = df["disease"].apply(map_disease)
    original = len(df)
    df = df[df["cnn_class"].notna()].copy()
    log.info(
        "Disease filter: %d → %d rows kept (%d classes matched)",
        original, len(df), df["cnn_class"].nunique()
    )
    if len(df) < 20:
        log.warning("Very few rows after filtering (%d). Synthetic examples will be added.", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Add synthetic examples for classes missing from CSV
# ══════════════════════════════════════════════════════════════════════════════
def add_synthetic(df: pd.DataFrame, cnn_classes: list) -> pd.DataFrame:
    covered = set(df["cnn_class"].unique()) if "cnn_class" in df.columns else set()
    rows = []
    for cls in cnn_classes:
        if cls not in covered:
            syns = SYNTHETIC_SYMPTOMS.get(cls, [])
            if syns:
                for text in syns:
                    rows.append({"cnn_class": cls, "text": text})
            else:
                # Minimal fallback from class name
                rows.append({"cnn_class": cls, "text": f"{cls.lower()} skin condition lesion rash"})
    if rows:
        syn_df = pd.DataFrame(rows)
        df = pd.concat([df, syn_df], ignore_index=True)
        log.info("Added %d synthetic rows for %d uncovered classes.", len(rows),
                 len(set(r["cnn_class"] for r in rows)))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SkinTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds       = out.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            preds       = out.logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load CNN classes ───────────────────────────────────────────────────
    cnn_classes = load_cnn_classes()
    if not cnn_classes:
        log.error("No CNN classes found. Run train_model.py first.")
        return

    # ── 2. Build dataframe ────────────────────────────────────────────────────
    csv_candidates = [
        args.csv,
        "disease_symptom_description.csv",
        "symptom_Description.csv",
        "symptom_precaution.csv",
        "dataset.csv",
        "Symptom-severity.csv",
    ]
    df = None
    for csv_path in csv_candidates:
        if csv_path and os.path.exists(csv_path):
            log.info("Loading symptom CSV: %s", csv_path)
            try:
                df = load_symptom_csv(csv_path)
                df = filter_and_map(df, cnn_classes)
                break
            except Exception as exc:
                log.warning("Failed to parse %s: %s", csv_path, exc)

    if df is None or len(df) == 0:
        log.warning(
            "No symptom CSV found or all rows filtered out.\n"
            "The text model will be trained on synthetic examples only.\n"
            "For better accuracy, download the Disease Symptom Description Dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset"
        )
        df = pd.DataFrame({"cnn_class": [], "text": []})

    # ── 3. Add synthetic examples ─────────────────────────────────────────────
    df = add_synthetic(df, cnn_classes)

    # ── 4. Build label map aligned with CNN classes ───────────────────────────
    # Label index MUST match CNN class index so fusion works correctly
    label_map   = {cls: idx for idx, cls in enumerate(cnn_classes)}
    df["label"] = df["cnn_class"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    log.info("Final dataset: %d rows, %d classes", len(df), df["cnn_class"].nunique())
    log.info("Class distribution:\n%s", df["cnn_class"].value_counts().to_string())

    # Save label map
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    log.info("Label map saved → %s", LABEL_MAP_PATH)

    # ── 5. Train/val split ────────────────────────────────────────────────────
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # Stratified split if enough samples, else simple split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.15, random_state=SEED, stratify=labels
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.15, random_state=SEED
        )

    log.info("Train: %d  Val: %d", len(X_train), len(X_val))

    # ── 6. Tokenizer + datasets ───────────────────────────────────────────────
    log.info("Loading BioBERT tokenizer: %s", BIOBERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)

    train_ds = SkinTextDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds   = SkinTextDataset(X_val,   y_val,   tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── 7. Model ──────────────────────────────────────────────────────────────
    n_classes = len(cnn_classes)
    log.info("Loading BioBERT for sequence classification (%d classes)...", n_classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        BIOBERT_MODEL,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── 8. Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── 9. Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = []

    log.info("═" * 55)
    log.info("BioBERT Training — %d epochs  batch=%d  lr=%g", args.epochs, args.batch, args.lr)
    log.info("═" * 55)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        vl_loss, vl_acc, vl_preds, vl_labels = eval_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": vl_loss, "val_acc": vl_acc})
        log.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f",
            epoch, args.epochs, tr_loss, tr_acc, vl_loss, vl_acc
        )
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_pretrained(TEXT_MODEL_DIR)
            tokenizer.save_pretrained(TEXT_MODEL_DIR)
            log.info("  ✓ Best model saved (val_acc=%.3f)", vl_acc)

    # ── 10. Final report ──────────────────────────────────────────────────────
    id2label = {idx: cls for cls, idx in label_map.items()}
    present_labels = sorted(set(vl_labels))
    present_names  = [id2label[i] for i in present_labels]
    report = classification_report(
        vl_labels, vl_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    log.info("\nValidation Classification Report:\n%s", report)

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "biobert_model":    BIOBERT_MODEL,
            "n_classes":        n_classes,
            "train_samples":    len(X_train),
            "val_samples":      len(X_val),
            "best_val_acc_pct": round(best_val_acc * 100, 2),
            "epochs":           args.epochs,
            "history":          history,
        }, f, indent=2)

    log.info("\n" + "═" * 55)
    log.info("TEXT MODEL TRAINING COMPLETE")
    log.info("  Best val accuracy : %.2f%%", best_val_acc * 100)
    log.info("  Saved → %s", TEXT_MODEL_DIR)
    log.info("  Label map → %s", LABEL_MAP_PATH)
    log.info("═" * 55)
    log.info("Next step: python app.py")


if __name__ == "__main__":
    main()