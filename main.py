import csv
import importlib.util
import inspect
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
CLEAN_CHUNKS_CSV = DATA_DIR / "cleaned_chunks.csv"
LABELLED_SENTIMENT_CSV = DATA_DIR / "labelled_sentiment.csv"
EMBEDDINGS_PATH = Path("embeddings.npy")
FAISS_INDEX_PATH = Path("hansard.index")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_MODEL = "bert-base-uncased"
GEN_MODEL = "google/flan-t5-base"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
MAX_SEQ_LEN = 256
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50

# To keep the assignment runnable on normal laptops
MAX_SENTIMENT_EXAMPLES = 900
BERT_EPOCHS = 1


# SECTION 1: Data Preparation and Embedding Generation
def extract_text(path: Path) -> str:
    """Extract text from one PDF using the fastest available local backend."""
    if pdfium is not None:
        return extract_text_pdfium(path)
    if PdfReader is not None:
        return extract_text_pypdf(path)
    if pdfplumber is not None:
        return extract_text_pdfplumber(path)
    raise RuntimeError(
        "No PDF text extraction backend is installed. Install pypdfium2, pypdf, "
        "or pdfplumber."
    )


def extract_text_pdfium(path: Path) -> str:
    text_parts = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            text_page = page.get_textpage()
            try:
                text_parts.append(text_page.get_text_range())
            finally:
                text_page.close()
                page.close()
    finally:
        pdf.close()
    return "\n".join(text_parts)


def extract_text_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_pdfplumber(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Remove common Hansard/PDF noise before chunking."""
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\bpage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"richfield graduate institute.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str) -> list[str]:
    """Split cleaned text into overlapping word-level windows."""
    words = text.split()
    if not words:
        return []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    return [
        " ".join(words[i : i + CHUNK_WORDS])
        for i in range(0, len(words), step)
        if len(words[i : i + CHUNK_WORDS]) >= 50
    ]


def read_cached_chunks(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    chunks = [row["text"] for row in rows if row.get("text")]
    print(f"[Data] Loaded {len(chunks)} cached chunks from {csv_path}.")
    return chunks


def write_chunks_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "source_file", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Data] Cleaned chunk dataset saved to {csv_path}.")


def load_corpus(data_dir: Path, use_cache: bool = True) -> list[str]:
    """Load Hansard PDFs, extract text, clean, chunk, and cache to CSV."""
    if use_cache and CLEAN_CHUNKS_CSV.exists():
        return read_cached_chunks(CLEAN_CHUNKS_CSV)

    pdfs = sorted(data_dir.glob("*.pdf"))
    rows = []
    print(f"[Data] Found {len(pdfs)} Hansard PDF files.")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")

    backend = (
        "pypdfium2"
        if pdfium is not None
        else "pypdf"
        if PdfReader is not None
        else "pdfplumber"
    )
    print(f"[Data] Extracting with {backend}. This may take a few minutes once.")

    start = time.perf_counter()
    for pdf_no, pdf_path in enumerate(pdfs, start=1):
        t0 = time.perf_counter()
        try:
            clean = clean_text(extract_text(pdf_path))
            pdf_chunks = chunk_text(clean)
        except Exception as exc:
            print(f"[Data] WARNING: skipped {pdf_path.name}: {exc}")
            continue

        for chunk in pdf_chunks:
            rows.append(
                {
                    "chunk_id": str(len(rows) + 1),
                    "source_file": pdf_path.name,
                    "text": chunk,
                }
            )
        print(
            f"[Data] {pdf_no:02d}/{len(pdfs)} {pdf_path.name}: "
            f"{len(pdf_chunks)} chunks in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    if not rows:
        raise RuntimeError("PDF extraction completed, but no usable text chunks were produced.")

    write_chunks_csv(rows, CLEAN_CHUNKS_CSV)
    print(f"[Data] {len(rows)} text chunks ready in {time.perf_counter() - start:.1f}s.")
    return [row["text"] for row in rows]


def demonstrate_tokenisation(sample: str) -> None:
    tokeniser = AutoTokenizer.from_pretrained(BERT_MODEL)
    tokens = tokeniser.tokenize(sample[:200])
    print(f"\n[Tokenisation] WordPiece sub-word tokens (first 20): {tokens[:20]}")
    print(f"[Tokenisation] Vocabulary size: {tokeniser.vocab_size:,}")


def generate_embeddings(chunks: list[str]) -> np.ndarray:
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] == len(chunks):
            print(f"[Embedding] Loaded cached embeddings from {EMBEDDINGS_PATH}: {embeddings.shape}")
            return embeddings.astype(np.float32)
        print("[Embedding] Cached embedding count does not match chunks; regenerating.")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"[Embedding] Saved {EMBEDDINGS_PATH}: {embeddings.shape} | Model: {EMBED_MODEL}")
    return embeddings


# SECTION 2: Fine-Tuning a Transformer Model for Sentiment
def pseudo_label(chunks: list[str]) -> list[int]:
    """Assign 3-class sentiment pseudo-labels using TextBlob polarity."""
    polarities = np.array([TextBlob(chunk).sentiment.polarity for chunk in chunks])
    p33, p67 = np.percentile(polarities, [33, 67])

    labels = []
    for polarity in polarities:
        if polarity <= p33:
            labels.append(LABEL_MAP["negative"])
        elif polarity >= p67:
            labels.append(LABEL_MAP["positive"])
        else:
            labels.append(LABEL_MAP["neutral"])

    counts = Counter(ID2LABEL[label] for label in labels)
    print(f"[Labels] Pseudo-label distribution: {dict(counts)}")
    return labels


def save_labelled_sentiment(chunks: list[str], labels: list[int]) -> None:
    rows = [
        {"chunk_id": i + 1, "label": ID2LABEL[label], "text": chunk}
        for i, (chunk, label) in enumerate(zip(chunks, labels))
    ]
    with LABELLED_SENTIMENT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "label", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Labels] Labelled sentiment dataset saved to {LABELLED_SENTIMENT_CSV}.")


def limit_training_examples(chunks: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    if MAX_SENTIMENT_EXAMPLES is None or len(chunks) <= MAX_SENTIMENT_EXAMPLES:
        return chunks, labels

    x_sample, _, y_sample, _ = train_test_split(
        chunks,
        labels,
        train_size=MAX_SENTIMENT_EXAMPLES,
        random_state=42,
        stratify=labels if can_stratify(labels) else None,
    )
    print(
        f"[BERT] Using {len(x_sample)} stratified chunks for fine-tuning "
        f"(set MAX_SENTIMENT_EXAMPLES=None to use all {len(chunks)})."
    )
    return x_sample, y_sample


def can_stratify(labels: list[int]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def split_sentiment_data(
    chunks: list[str], labels: list[int]
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    stratify = labels if can_stratify(labels) else None
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        chunks,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )
    stratify_tmp = y_tmp if can_stratify(y_tmp) else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.50,
        random_state=42,
        stratify=stratify_tmp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_hf_dataset(texts: list[str], labels: list[int], tokeniser) -> Dataset:
    """Tokenise texts and return a PyTorch-formatted HuggingFace Dataset."""
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tokenise(batch):
        return tokeniser(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )

    ds = ds.map(tokenise, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_training_args() -> TrainingArguments:
    kwargs = {
        "output_dir": "bert_sentiment_ckpt",
        "num_train_epochs": BERT_EPOCHS,
        "per_device_train_batch_size": 16 if torch.cuda.is_available() else 8,
        "per_device_eval_batch_size": 32,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_steps": 50,
        "report_to": "none",
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


class ManualTorchTrainer:

    def __init__(self, model) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_ds: Dataset, val_ds: Dataset) -> None:
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        for epoch in range(BERT_EPOCHS):
            self.model.train()
            losses = []
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimiser.zero_grad()
                output = self.model(**batch)
                output.loss.backward()
                optimiser.step()
                losses.append(float(output.loss.detach().cpu()))

            metrics = compute_metrics_from_predictions(self.predict(val_ds))
            print(
                f"[BERT] Epoch {epoch + 1}/{BERT_EPOCHS} "
                f"loss={np.mean(losses):.4f} val_f1={metrics['f1']:.4f}"
            )

    def predict(self, dataset: Dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=32)
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"].numpy()
                inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                    if key != "labels"
                }
                output = self.model(**inputs)
                all_logits.append(output.logits.detach().cpu().numpy())
                all_labels.append(labels)

        return SimpleNamespace(
            predictions=np.concatenate(all_logits, axis=0),
            label_ids=np.concatenate(all_labels, axis=0),
        )


def compute_metrics_from_predictions(prediction_output) -> dict:
    preds = np.argmax(prediction_output.predictions, axis=-1)
    return compute_metrics((prediction_output.predictions, prediction_output.label_ids)) | {
        "preds": preds
    }


def fine_tune_bert(train_ds: Dataset, val_ds: Dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    if importlib.util.find_spec("accelerate") is None:
        print("[BERT] accelerate is not installed; using a lightweight PyTorch trainer.")
        trainer = ManualTorchTrainer(model)
        trainer.train(train_ds, val_ds)
        return trainer

    trainer = Trainer(
        model=model,
        args=make_training_args(),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    print("[BERT] Training started.")
    trainer.train()
    return trainer


def evaluate_bert(trainer: Trainer, test_ds: Dataset) -> None:
    """Print classification metrics and save a confusion matrix image."""
    out = trainer.predict(test_ds)
    preds = np.argmax(out.predictions, axis=-1)
    true = out.label_ids

    print("\n[BERT Evaluation] Classification Report:")
    print(classification_report(true, preds, target_names=["negative", "neutral", "positive"]))

    cm = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("BERT Policy Sentiment - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_sentiment.png", dpi=150)
    plt.close(fig)
    print("[BERT Evaluation] Confusion matrix saved to confusion_matrix_sentiment.png")


# SECTION 3: Retrieval-Augmented Generation (RAG)
class RAGSystem:
    """FAISS retriever plus FLAN-T5 generator for grounded policy QA."""

    def __init__(self, chunks: list[str], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.build_index(embeddings)
        self.load_generator()

    def build_index(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        print(
            f"[RAG] FAISS index saved to {FAISS_INDEX_PATH}: "
            f"{self.index.ntotal} vectors, dim={dim}."
        )

    def load_generator(self) -> None:
        print(f"[RAG] Loading generator '{GEN_MODEL}'.")
        self.tokeniser = AutoTokenizer.from_pretrained(GEN_MODEL)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
        self.generator.eval()

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-k chunks most semantically similar to the query."""
        top_k = min(top_k, self.index.ntotal)
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        return [
            (self.chunks[i], float(scores[0][rank]))
            for rank, i in enumerate(idxs[0])
            if i >= 0
        ]

    def generate(self, query: str, hits: list[tuple[str, float]]) -> str:
        """Generate a grounded answer from retrieved context using FLAN-T5."""
        context = " ".join(chunk for chunk, _ in hits)[:1800]
        prompt = (
            "Answer the question using only the South African parliamentary "
            "debate context below. If the context is insufficient, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        inputs = self.tokeniser(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            ids = self.generator.generate(
                **inputs,
                max_new_tokens=160,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokeniser.decode(ids[0], skip_special_tokens=True)

    def query(self, question: str, top_k: int = 5) -> dict[str, object]:
        """Run retrieval, generation, and print source evidence."""
        hits = self.retrieve(question, top_k)
        answer = self.generate(question, hits)
        top_score = hits[0][1] if hits else 0.0
        top_text = hits[0][0][:180] if hits else "No evidence retrieved."
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"[Top evidence] score={top_score:.4f}: {top_text}...")
        return {"question": question, "answer": answer, "hits": hits}


def evaluate_rag_responses(results: list[dict[str, object]]) -> None:
    """Print a simple relevance and grounding evaluation table."""
    print("\n[RAG Evaluation]")
    print("Query | Retrieved relevant? | Grounded? | Score")
    print("-" * 72)
    for result in results:
        hits = result["hits"]
        answer = str(result["answer"]).strip()
        relevant = "Yes" if hits and hits[0][1] >= 0.20 else "Review"
        grounded = "Yes" if answer and answer.lower() != "i don't know" else "Review"
        score = "5/5" if relevant == "Yes" and grounded == "Yes" else "3/5"
        print(f"{result['question']} | {relevant} | {grounded} | {score}")


# SECTION 4: Ethics, Risk, and Responsible AI answer is found in the assignment pdf "402412737_MC

def main() -> None:
    print("=" * 66)
    print("  COMPONENT B - LLM & RAG | South African Parliamentary Hansard")
    print("=" * 66)

    print("\nSECTION 1: Data Preparation and Embedding Generation")
    chunks = load_corpus(DATA_DIR)
    demonstrate_tokenisation(chunks[0])
    embeddings = generate_embeddings(chunks)

    print("\nSECTION 2: Fine-Tuning BERT for Policy Sentiment Classification")
    labels = pseudo_label(chunks)
    save_labelled_sentiment(chunks, labels)
    train_chunks, train_labels = limit_training_examples(chunks, labels)
    x_train, x_val, x_test, y_train, y_val, y_test = split_sentiment_data(
        train_chunks, train_labels
    )

    tokeniser = AutoTokenizer.from_pretrained(BERT_MODEL)
    train_ds = build_hf_dataset(x_train, y_train, tokeniser)
    val_ds = build_hf_dataset(x_val, y_val, tokeniser)
    test_ds = build_hf_dataset(x_test, y_test, tokeniser)

    trainer = fine_tune_bert(train_ds, val_ds)
    evaluate_bert(trainer, test_ds)

    print("\nSECTION 3: Retrieval-Augmented Generation (RAG)")
    rag = RAGSystem(chunks, embeddings)

    print("\n[RAG Demo Queries]")
    rag_results = []
    for question in [
        "What was the sentiment on education policy in 2023?",
        "How did parliament address unemployment and job creation?",
        "What was discussed regarding healthcare budget allocation?",
    ]:
        rag_results.append(rag.query(question))
    evaluate_rag_responses(rag_results)

    print("Component B pipeline complete.")


if __name__ == "__main__":
    main()
