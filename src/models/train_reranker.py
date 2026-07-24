"""
src/models/train_reranker.py
=============================
G.G.A Takımı — BAAI/bge-reranker-large Fine-Tuning & Evaluator Pipeline

Özellikler:
1. Backbone Model: BAAI/bge-reranker-large (Varsayılan) veya dbmdz/bert-base-turkish-cased
2. Training: PyTorch / HuggingFace AutoModelForSequenceClassification veya CrossEncoder
3. Hyperparameters: Max length=256, Batch size=16 (Grad Accum=2), LR=2e-5 (Warmup + Cosine), Epochs=3-5, fp16/bf16
4. Evaluation: MRR@10 ve NDCG@10 metriklerini hesaplayan Custom Reranker Evaluator
5. Save: model.save_pretrained(output_dir)
"""

from __future__ import annotations

import math
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_cosine_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# 1. Dataset & Data Loader
# =============================================================================

class RerankerDataset(Dataset):
    """
    Cross-Encoder için (Query, Product Document, Label) çiftleri üreten Dataset.
    Triplets formatından binary classification örnekleri türetir.
    """
    def __init__(self, triplets: list[dict[str, Any]], tokenizer: Any, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[tuple[str, str, float]] = []

        for trip in triplets:
            q = trip["query"]
            pos = trip["positive_doc"]
            negs = trip.get("hard_negatives", [])

            # Pozitif örnek (label = 1.0)
            self.examples.append((q, pos, 1.0))

            # Hard negatif örnekler (label = 0.0)
            for neg in negs:
                self.examples.append((q, neg, 0.0))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        q, doc, label = self.examples[idx]
        inputs = self.tokenizer(
            q,
            doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


# =============================================================================
# 2. Evaluation Metrics (MRR@10 & NDCG@10)
# =============================================================================

def compute_mrr_at_k(eval_samples: list[dict[str, Any]], k: int = 10) -> float:
    """
    Mean Reciprocal Rank (MRR@k) Hesablar.
    eval_samples: List of {'query': q, 'candidates': [(doc, is_relevant, score), ...]}
    """
    reciprocal_ranks = []
    for sample in eval_samples:
        sorted_candidates = sorted(sample["candidates"], key=lambda x: x[2], reverse=True)[:k]
        found_rank = 0
        for rank, (_, is_rel, _) in enumerate(sorted_candidates, 1):
            if is_rel:
                found_rank = rank
                break
        reciprocal_ranks.append(1.0 / found_rank if found_rank > 0 else 0.0)
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if eval_samples else 0.0


def compute_ndcg_at_k(eval_samples: list[dict[str, Any]], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@k) Hesaplar.
    """
    ndcg_list = []
    for sample in eval_samples:
        sorted_candidates = sorted(sample["candidates"], key=lambda x: x[2], reverse=True)[:k]
        
        # DCG
        dcg = 0.0
        for rank, (_, is_rel, _) in enumerate(sorted_candidates, 1):
            if is_rel:
                dcg += 1.0 / math.log2(rank + 1)
                
        # Ideal DCG (IDCG)
        total_rel = sum(1 for _, is_rel, _ in sample["candidates"] if is_rel)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(total_rel, k) + 1))
        
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return float(sum(ndcg_list) / len(ndcg_list)) if eval_samples else 0.0


def evaluate_reranker(
    model: Any,
    tokenizer: Any,
    eval_triplets: list[dict[str, Any]],
    device: torch.device,
    max_length: int = 256,
    k: int = 10
) -> dict[str, float]:
    """
    Reranker modeli üzerinde MRR@10 ve NDCG@10 metriklerini hesaplar.
    """
    model.eval()
    eval_samples = []

    with torch.no_grad():
        for trip in eval_triplets:
            q = trip["query"]
            pos = trip["positive_doc"]
            negs = trip.get("hard_negatives", [])

            all_docs = [(pos, 1)] + [(neg, 0) for neg in negs]
            candidates = []

            for doc, is_rel in all_docs:
                inputs = tokenizer(
                    q,
                    doc,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1)
                score = float(torch.sigmoid(logits).cpu().item()) if logits.ndim == 0 else float(torch.sigmoid(logits)[0].cpu().item())
                candidates.append((doc, is_rel, score))

            eval_samples.append({"query": q, "candidates": candidates})

    mrr = compute_mrr_at_k(eval_samples, k=k)
    ndcg = compute_ndcg_at_k(eval_samples, k=k)
    return {"mrr@10": mrr, "ndcg@10": ndcg}


# =============================================================================
# 3. Training Pipeline
# =============================================================================

def train_reranker_model(
    train_triplets: list[dict[str, Any]],
    eval_triplets: list[dict[str, Any]] | None = None,
    model_name: str = "BAAI/bge-reranker-large",
    output_dir: str = "outputs/reranker_model",
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    use_fp16: bool = True,
) -> dict[str, Any]:
    """
    BAAI/bge-reranker-large modelini fine-tune eden ana eğitim fonksiyonu.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers kütüphanesi yüklü değil: pip install transformers torch")

    print(f"[+] Cross-Encoder Backbone Yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = RerankerDataset(train_triplets, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and torch.cuda.is_available())

    print(f"[+] Fine-Tuning Başladı: {epochs} Epoch, Batch Size={batch_size}, LR={learning_rate}, Device={device}")
    start_time = time.time()

    best_mrr = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            token_type_ids = batch.get("token_type_ids")
            kwargs = {"token_type_ids": token_type_ids.to(device)} if token_type_ids is not None else {}

            with torch.cuda.amp.autocast(enabled=use_fp16 and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch [{epoch}/{epochs}] — Train Loss: {avg_loss:.4f}")

        # Evaluation
        if eval_triplets:
            metrics = evaluate_reranker(model, tokenizer, eval_triplets, device, max_length=max_length)
            print(f"  Epoch [{epoch}/{epochs}] — MRR@10: {metrics['mrr@10']:.4f} | NDCG@10: {metrics['ndcg@10']:.4f}")

            if metrics["mrr@10"] >= best_mrr:
                best_mrr = metrics["mrr@10"]
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"  [★] En iyi model kaydedildi: {output_dir}")

    if not eval_triplets:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    print(f"[+] Reranker Fine-Tuning Tamamlandı ({elapsed:.2f} saniye)")

    return {
        "model_dir": output_dir,
        "elapsed_seconds": elapsed,
        "best_mrr_at_10": best_mrr if eval_triplets else None
    }
