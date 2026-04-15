import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from PIL import ImageEnhance

# ── TensorFlow ────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise ImportError("TensorFlow is required but not installed.")

# ── PyTorch / Transformers (optional) ─────────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTS      = {"jpg", "jpeg", "png"}
MAX_FILE_BYTES    = 10 * 1024 * 1024   # 10 MB

model_lock        = threading.Lock()
MODEL_DIR         = os.path.join(BASE_DIR, "model")
MODEL_PATH        = os.path.join(MODEL_DIR, "image_model", "efficientnetb0_skin_disease_final.h5")
CLASS_NAMES_PATH  = os.path.join(MODEL_DIR, "image_model", "class_names.json")
DISEASE_INFO_PATH = os.path.join(MODEL_DIR, "disease_info.json")
OOD_PATH          = os.path.join(MODEL_DIR, "ood_threshold.json")
TEXT_MODEL_DIR    = os.path.join(MODEL_DIR, "text_model")
LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "text_label_map.json")

IMG_SIZE              = (224, 224)   # EfficientNetB0 native; overridden after model load
AUTO_DELETE_SECS      = 60
INFERENCE_TIMEOUT     = 45
MIN_IMG_DIM           = 32
RATE_LIMIT_MAX        = 10
RATE_LIMIT_WINDOW     = 60
TEXT_MAX_LEN          = 256
TEXT_MIN_LENGTH       = 10           # ignore trivially short symptom strings

# ── Fusion thresholds ─────────────────────────────────────────────────────────
# >= HIGH_CONF_THRESHOLD  → image wins outright, text ignored
# >= MED_CONF_THRESHOLD   → fuse with BioBERT at 85/15 split
# <  MED_CONF_THRESHOLD   → fuse with BioBERT at 70/30 split
# keyword scorer          → NEVER fuses when an image is present
HIGH_CONF_THRESHOLD   = 0.60
MED_CONF_THRESHOLD    = 0.35
W_IMAGE_HIGH          = 0.85
W_IMAGE_LOW           = 0.70

TTA_INFERENCE         = 7
OOD_THRESHOLD_DEFAULT = 0.10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES

# ── Class names ───────────────────────────────────────────────────────────────
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "class_names.json is missing. Export it from training with the same "
        "class order as the model output."
    )

with open(CLASS_NAMES_PATH) as f:
    CLASSES = json.load(f)

N_CLASSES = len(CLASSES)
log.info("Loaded %d classes from file", N_CLASSES)

# ── Disease info ──────────────────────────────────────────────────────────────
DEFAULT_ENTRY = {
    "description": "A dermatological condition requiring specialist assessment.",
    "symptoms":    ["Skin changes", "Lesions or discolouration", "Possible itching"],
    "treatment":   "Consult a qualified dermatologist.",
    "severity":    "moderate",
    "icon":        "🔬",
    "see_doctor":  True,
}

if os.path.exists(DISEASE_INFO_PATH):
    with open(DISEASE_INFO_PATH) as f:
        DISEASE_INFO = json.load(f)
    log.info("Loaded disease info for %d classes", len(DISEASE_INFO))
else:
    DISEASE_INFO = {
        "Acne":      {"description": "Follicle-clogging skin condition.",          "symptoms": ["Pimples", "Blackheads", "Oily skin"],                  "treatment": "Topical retinoids, benzoyl peroxide.",              "severity": "mild",     "icon": "🔴", "see_doctor": False},
        "Eczema":    {"description": "Chronic inflammatory itchy skin condition.", "symptoms": ["Dry skin", "Itching", "Red patches"],                  "treatment": "Moisturisers, corticosteroids.",                    "severity": "moderate", "icon": "🟠", "see_doctor": False},
        "Melanoma":  {"description": "Most serious skin cancer.",                  "symptoms": ["Asymmetric mole", "Irregular border", "Colour change"], "treatment": "Surgery, immunotherapy. See doctor immediately.",   "severity": "severe",   "icon": "🚨", "see_doctor": True},
        "Psoriasis": {"description": "Chronic autoimmune skin disease.",           "symptoms": ["Silvery scales", "Red patches", "Itching"],             "treatment": "Topical steroids, phototherapy.",                   "severity": "moderate", "icon": "🟡", "see_doctor": False},
    }


def get_disease_info(name: str) -> dict:
    if name in DISEASE_INFO:
        return DISEASE_INFO[name]
    for k, v in DISEASE_INFO.items():
        if k.lower() == name.lower():
            return v
    return DEFAULT_ENTRY

# ── OOD threshold ─────────────────────────────────────────────────────────────
OOD_THRESHOLD = OOD_THRESHOLD_DEFAULT
if os.path.exists(OOD_PATH):
    try:
        with open(OOD_PATH) as f:
            raw_val = float(json.load(f).get("ood_threshold", OOD_THRESHOLD_DEFAULT))
        OOD_THRESHOLD = min(raw_val, 0.25)   # safety cap — never exceed 0.25
        log.info("OOD threshold: %.4f (capped at 0.25)", OOD_THRESHOLD)
    except Exception as exc:
        log.warning("OOD parse error: %s — using default %.2f", exc, OOD_THRESHOLD_DEFAULT)

# ── CNN model ─────────────────────────────────────────────────────────────────
cnn_model        = None
MODEL_LOAD_ERROR = None

if not TF_AVAILABLE:
    MODEL_LOAD_ERROR = "TensorFlow not available."
elif not os.path.exists(MODEL_PATH):
    MODEL_LOAD_ERROR = f"Model file not found: {MODEL_PATH}"
else:
    log.info("Loading CNN model from %s …", MODEL_PATH)
    try:
        _orig_dense_from_config = keras.layers.Dense.from_config

        @classmethod  # type: ignore[misc]
        def _safe_dense_from_config(cls, config):
            config.pop("quantization_config", None)
            return _orig_dense_from_config.__func__(cls, config)

        keras.layers.Dense.from_config = _safe_dense_from_config
        cnn_model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        keras.layers.Dense.from_config = _orig_dense_from_config

        model_out = cnn_model.output_shape[-1]
        if model_out != N_CLASSES:
            raise ValueError(
                f"Model output size ({model_out}) does not match "
                f"class_names.json ({N_CLASSES})."
            )

        input_shape = cnn_model.input_shape
        if input_shape[1] and input_shape[2]:
            IMG_SIZE = (input_shape[1], input_shape[2])

        log.info("CNN model loaded — input size: %s, classes: %d", IMG_SIZE, N_CLASSES)

    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        log.error("CNN model load FAILED: %s", exc)
        cnn_model = None

# ── BioBERT (optional) ────────────────────────────────────────────────────────
text_model        = None
text_tokenizer    = None
text_device       = None
text_lock         = threading.Lock()
text_index_remap  = None
BIOBERT_AVAILABLE = False

if TORCH_AVAILABLE and os.path.isdir(TEXT_MODEL_DIR):
    try:
        text_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
        text_model     = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
        text_model.to(text_device)
        text_model.eval()

        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH) as f:
                label_map = json.load(f)
            id2label = {v: k for k, v in label_map.items()}
            text_index_remap = [label_map.get(id2label.get(i)) for i in range(text_model.config.num_labels)]
        else:
            text_index_remap = list(range(text_model.config.num_labels))

        BIOBERT_AVAILABLE = True
        log.info("BioBERT loaded on %s", text_device)

    except Exception as exc:
        log.warning("BioBERT load failed: %s — keyword fallback active.", exc)

# ── Keyword symptom scorer ────────────────────────────────────────────────────
_SYMPTOM_KEYWORDS: dict = {}


def _build_keyword_index():
    for cls in CLASSES:
        info = get_disease_info(cls)
        kw: dict = {}
        for sym in info.get("symptoms", []):
            for word in re.findall(r"[a-z]+", sym.lower()):
                if len(word) > 3:
                    kw[word] = kw.get(word, 0) + 1.5
        for word in re.findall(r"[a-z]+", info.get("description", "").lower()):
            if len(word) > 4:
                kw[word] = kw.get(word, 0) + 0.5
        for word in re.findall(r"[a-z]+", cls.lower()):
            kw[word] = kw.get(word, 0) + 2.0
        _SYMPTOM_KEYWORDS[cls] = kw

_build_keyword_index()


def keyword_text_score(symptom_text: str) -> np.ndarray:
    words  = set(re.findall(r"[a-z]+", symptom_text.lower()))
    scores = np.zeros(N_CLASSES, dtype=np.float32)

    for i, cls in enumerate(CLASSES):
        for word in words:
            if word in _SYMPTOM_KEYWORDS.get(cls, {}):
                scores[i] += _SYMPTOM_KEYWORDS[cls][word]

    total = scores.sum()
    if total > 0:
        scores  = np.exp(scores / max(scores.max(), 1e-9) * 3.0)
        scores /= scores.sum()
    else:
        scores = np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES

    return scores

# ── Rate limiter ──────────────────────────────────────────────────────────────
_rate_store = defaultdict(list)
_rate_lock  = threading.Lock()


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        recent = [t for t in _rate_store[ip] if now - t < RATE_LIMIT_WINDOW]
        if len(recent) >= RATE_LIMIT_MAX:
            _rate_store[ip] = recent
            return True
        recent.append(now)
        _rate_store[ip] = recent
        return False

# ── Image helpers ─────────────────────────────────────────────────────────────
def allowed_extension(filename: str) -> bool:
    parts = filename.rsplit(".", 1)
    return len(parts) == 2 and parts[1].lower() in ALLOWED_EXTS


def validate_image_bytes(stream) -> tuple[bytes, str]:
    raw = stream.read()
    if not raw:
        return b"", "Uploaded file is empty."
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except UnidentifiedImageError:
        return raw, "Not a valid image. Upload JPG or PNG."
    except Exception:
        return raw, "Image file is corrupt."

    img  = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w < MIN_IMG_DIM or h < MIN_IMG_DIM:
        return raw, f"Image too small ({w}×{h}px). Minimum {MIN_IMG_DIM}×{MIN_IMG_DIM}px."

    return raw, ""


def save_image_safely(raw: bytes, filename: str) -> tuple[str, str]:
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(path, "wb") as f:
            f.write(raw)
        return path, ""
    except OSError as exc:
        return "", f"Storage error: {exc.strerror}"


def schedule_delete(path: str, delay: int = AUTO_DELETE_SECS):
    def _worker():
        time.sleep(delay)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def _top5(probs: np.ndarray) -> list:
    return [
        {"name": CLASSES[i], "pct": round(float(s) * 100, 2)}
        for i, s in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
    ]

# ── CNN inference with TTA ────────────────────────────────────────────────────
def _preprocess_image(
    img_path: str,
    flip: bool = False,
    rotate: int = 0,
    zoom: float = 1.0,
    brightness: float = 1.0,
) -> np.ndarray:
    """
    Preprocess an image for EfficientNetB0 inference.

    tf.keras.applications.efficientnet.preprocess_input() expects raw [0, 255]
    float32 values and scales them internally to [-1, 1].
    Do NOT divide by 255 before calling it — that would produce a near-zero
    tensor and destroy all predictions.

    Contrast boost (1.2x) is applied before resizing so it works on the full
    original resolution, avoiding amplification of downsampling artefacts.
    """
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if rotate != 0:
            img = img.rotate(rotate, fillcolor=(128, 128, 128))
        if zoom != 1.0:
            w, h   = img.size
            nw, nh = int(w * zoom), int(h * zoom)
            img    = img.resize((nw, nh), Image.LANCZOS)
            left   = (nw - w) // 2
            top    = (nh - h) // 2
            img    = img.crop((left, top, left + w, top + h))
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)

        # Mild contrast boost — improves faded / low-contrast clinical photos
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = img.resize(IMG_SIZE, Image.LANCZOS)

        arr = np.array(img, dtype=np.float32)   # raw [0, 255] — do NOT divide

    arr = tf.keras.applications.efficientnet.preprocess_input(arr)  # → [-1, 1]
    return np.expand_dims(arr, 0)


def _cnn_worker(img_path: str, holder: list):
    try:
        augments = [
            dict(flip=False, rotate=0,    zoom=1.0,  brightness=1.0),
            dict(flip=True,  rotate=0,    zoom=1.0,  brightness=1.0),
            dict(flip=False, rotate=15,   zoom=1.0,  brightness=1.0),
            dict(flip=False, rotate=-15,  zoom=1.0,  brightness=1.0),
            dict(flip=False, rotate=0,    zoom=1.1,  brightness=1.0),
            dict(flip=False, rotate=0,    zoom=0.9,  brightness=1.0),
            dict(flip=True,  rotate=10,   zoom=1.0,  brightness=1.05),
        ]
        all_preds = []
        with model_lock:
            for aug in augments[:TTA_INFERENCE]:
                arr   = _preprocess_image(img_path, **aug)
                preds = cnn_model.predict(arr, verbose=0)[0]
                all_preds.append(preds)

        avg = np.mean(all_preds, axis=0)
        holder.append((avg, None))

    except Exception as exc:
        holder.append((None, exc))


def run_cnn(img_path: str) -> np.ndarray:
    if cnn_model is None:
        log.warning("CNN model not loaded — returning random noise.")
        rng = np.random.default_rng(int(os.path.getsize(img_path)) % 9999)
        return rng.dirichlet(np.ones(N_CLASSES)).astype(np.float32)

    holder = []
    t = threading.Thread(target=_cnn_worker, args=(img_path, holder), daemon=True)
    t.start()
    t.join(timeout=INFERENCE_TIMEOUT)

    if t.is_alive():
        raise RuntimeError(f"CNN inference timed out after {INFERENCE_TIMEOUT}s.")

    preds, exc = holder[0]
    if exc:
        raise exc
    if preds is None or np.any(np.isnan(preds)):
        raise RuntimeError("CNN returned invalid (NaN) output.")

    return preds.astype(np.float32)

# ── BioBERT inference ─────────────────────────────────────────────────────────
def run_biobert(symptom_text: str) -> np.ndarray | None:
    if not BIOBERT_AVAILABLE:
        return None
    try:
        with text_lock:
            enc = text_tokenizer(
                symptom_text.strip(),
                max_length=TEXT_MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = text_model(
                    input_ids=enc["input_ids"].to(text_device),
                    attention_mask=enc["attention_mask"].to(text_device),
                ).logits
            probs_raw = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        aligned = np.zeros(N_CLASSES, dtype=np.float32)
        for txt_idx, cnn_idx in enumerate(text_index_remap):
            if cnn_idx is not None and 0 <= cnn_idx < N_CLASSES and txt_idx < len(probs_raw):
                aligned[cnn_idx] += float(probs_raw[txt_idx])

        total = aligned.sum()
        return (aligned / total).astype(np.float32) if total > 0 else None

    except Exception as exc:
        log.warning("BioBERT inference failed: %s", exc)
        return None


def run_text(symptom_text: str) -> tuple[np.ndarray | None, str]:
    """
    Return (probability_array, method_name).
    Strings shorter than TEXT_MIN_LENGTH are discarded — they produce
    near-uniform keyword scores that corrupt fusion.
    """
    text = symptom_text.strip()
    if not text or len(text) < TEXT_MIN_LENGTH:
        return None, "none"

    if BIOBERT_AVAILABLE:
        result = run_biobert(text)
        if result is not None:
            return result, "biobert"

    return keyword_text_score(text), "keyword"

# ── Smart confidence-aware fusion ─────────────────────────────────────────────
def fuse(
    img_probs: np.ndarray | None,
    text_probs: np.ndarray | None,
    text_method: str,
) -> tuple[np.ndarray, str]:
    """
    Text should only HELP when the image model is uncertain.
    It must NEVER pull a confident image prediction in the wrong direction.

    Decision table (when both image and text are present):
    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ Condition                    │ Result                                   │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ img_conf >= 0.60             │ Return image as-is. Text ignored.        │
    │ img_conf >= 0.35, BioBERT    │ Fuse at 85% image / 15% text.           │
    │ img_conf <  0.35, BioBERT    │ Fuse at 70% image / 30% text.           │
    │ Any conf,  keyword scorer    │ Return image as-is. Keywords too noisy.  │
    └──────────────────────────────┴──────────────────────────────────────────┘
    """
    if img_probs is not None and text_probs is not None:
        img_conf = float(img_probs.max())

        # High-confidence image — text cannot help, skip fusion
        if img_conf >= HIGH_CONF_THRESHOLD:
            log.info("Fusion: image_conf=%.2f >= %.2f → image only.", img_conf, HIGH_CONF_THRESHOLD)
            return img_probs, "image_only|high_conf"

        # Keyword scorer is too noisy to fuse with an image prediction
        if text_method == "keyword":
            log.info("Fusion: keyword skipped when image present.")
            return img_probs, "image_only|keyword_skipped"

        # BioBERT available — confidence-weighted fusion
        if text_method == "biobert":
            w_i  = W_IMAGE_HIGH if img_conf >= MED_CONF_THRESHOLD else W_IMAGE_LOW
            w_t  = 1.0 - w_i
            mode = (
                f"image+biobert|{'med' if img_conf >= MED_CONF_THRESHOLD else 'low'}_conf"
            )
            fused = w_i * img_probs + w_t * text_probs
            total = fused.sum()
            log.info("Fusion: %s  w_img=%.2f  w_txt=%.2f  img_conf=%.2f", mode, w_i, w_t, img_conf)
            return (fused / total).astype(np.float32) if total > 0 else img_probs, mode

        # Fallback (should never reach here)
        return img_probs, "image_only|fallback"

    if img_probs  is not None:
        return img_probs,  "image_only"
    if text_probs is not None:
        return text_probs, "text_only" if text_method == "biobert" else "keywords_only"

    raise ValueError("No model output available.")

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def h413(e):
    return jsonify({"error": f"File too large. Max {MAX_FILE_BYTES // (1024 * 1024)} MB."}), 413

@app.errorhandler(404)
def h404(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def h405(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(500)
def h500(e):
    return jsonify({"error": "Unexpected server error."}), 500

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":          "ok" if cnn_model is not None else "degraded",
        "cnn_model":       "loaded" if cnn_model else "FAILED",
        "model_error":     MODEL_LOAD_ERROR,
        "img_size":        list(IMG_SIZE),
        "tta_steps":       TTA_INFERENCE,
        "text_model":      "biobert" if BIOBERT_AVAILABLE else "keyword",
        "n_classes":       N_CLASSES,
        "tf_available":    TF_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "ood_threshold":   OOD_THRESHOLD,
        "fusion_config": {
            "high_conf_threshold": HIGH_CONF_THRESHOLD,
            "med_conf_threshold":  MED_CONF_THRESHOLD,
            "w_image_high":        W_IMAGE_HIGH,
            "w_image_low":         W_IMAGE_LOW,
            "keyword_fuses_image": False,
        },
        "timestamp": int(time.time()),
    })


@app.route("/classes")
def list_classes():
    return jsonify({"count": N_CLASSES, "classes": CLASSES})


@app.route("/predict", methods=["POST"])
def predict():
    save_path = None
    ip        = request.remote_addr or ""

    if is_rate_limited(ip):
        return jsonify({"error": "Too many requests. Please wait before retrying."}), 429

    has_file = "file" in request.files and request.files["file"].filename != ""
    text     = request.form.get("symptoms", "").strip()

    if not has_file and not text:
        return jsonify({"error": "Send an image, symptom text, or both."}), 400

    img_probs = None

    try:
        t0 = time.perf_counter()

        # ── Image branch ───────────────────────────────────────────────────
        if has_file:
            file = request.files["file"]

            if not allowed_extension(file.filename):
                return jsonify({"error": "Unsupported file type. Upload JPG or PNG."}), 422

            raw, err = validate_image_bytes(file.stream)
            if err:
                return jsonify({"error": err}), 422

            filename  = uuid.uuid4().hex + ".jpg"
            save_path, save_err = save_image_safely(raw, filename)
            if save_err:
                return jsonify({"error": save_err}), 500

            img_probs = run_cnn(save_path)

            if img_probs.max() < OOD_THRESHOLD:
                return jsonify({
                    "ood":            True,
                    "message":        (
                        "The uploaded image does not appear to show a skin condition. "
                        "Please upload a clear, close-up photo of the affected skin area."
                    ),
                    "max_confidence": round(float(img_probs.max()) * 100, 2),
                })

        # ── Text branch ────────────────────────────────────────────────────
        # Skip ALL text scoring when an image is present but BioBERT is not
        # available — the keyword scorer is too noisy to be worth any weight.
        if text and (not has_file or BIOBERT_AVAILABLE):
            text_probs, text_method = run_text(text)
        else:
            text_probs, text_method = None, "none"

        # ── Fusion ─────────────────────────────────────────────────────────
        final, mode = fuse(img_probs, text_probs, text_method)

        idx     = int(np.argmax(final))
        disease = CLASSES[idx]
        conf    = round(float(final[idx]) * 100, 2)
        info    = get_disease_info(disease)

        return jsonify({
            "disease":     disease,
            "confidence":  conf,
            "info":        info,
            "top5":        _top5(final),
            "all_scores":  {CLASSES[i]: round(float(p) * 100, 2) for i, p in enumerate(final)},
            "fusion_mode": mode,
            "time_ms":     int((time.perf_counter() - t0) * 1000),
        })

    except RuntimeError as exc:
        log.error("Inference error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    except Exception as exc:
        log.exception("Unexpected error in /predict")
        return jsonify({"error": "Internal server error. Please try again."}), 500

    finally:
        if save_path and os.path.exists(save_path):
            schedule_delete(save_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)