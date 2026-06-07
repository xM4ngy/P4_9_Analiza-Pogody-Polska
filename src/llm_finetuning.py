"""Część LLM: przygotowanie datasetu, fine-tuning i ewaluacja allegro/plt5-small.

Skrypt nie jest uruchamiany domyślnie, ponieważ trening wymaga internetu oraz najlepiej GPU.
Przykłady:
    python main.py --mode llm-dataset
    python src/llm_finetuning.py --train --epochs 50 --lr 3e-4
    python src/llm_finetuning.py --generate            # po treningu: przykłady generacji

Po treningu zapisywane są:
    outputs/llm_eval.json        - metryki (eval_loss, BLEU, ROUGE) + przykłady generacji
    outputs/plt5-weather/        - wagi modelu i tokenizer (duże, w .gitignore)

Uwaga o hiperparametrach:
    allegro/plt5-small (baza mT5) na małym, szablonowym zbiorze wymaga wyższego learning rate
    (3e-4 .. 1e-3) i większej liczby epok (>=50). Zbyt niski LR (np. 5e-5) prowadzi do
    niedouczenia i degeneracyjnych generacji (powtórzenia, błędne tokeny).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "allegro/plt5-small"
DEFAULT_INPUT = "data/pogoda_polska.csv"
DEFAULT_DATASET = "data/llm_dataset.csv"
DEFAULT_MODEL_DIR = "outputs/plt5-weather"
DEFAULT_EVAL_PATH = "outputs/llm_eval.json"
DEFAULT_LR = 3e-4
GEN_MAX_LEN = 160
NO_REPEAT_NGRAM = 3


def _num(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_dataset(input_csv: str = DEFAULT_INPUT, output_csv: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Tworzy pary input_text -> target_text na podstawie danych pogodowych."""
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = ["Miasto", "Temperatura", "Temp_Odczuwalna", "Cisnienie", "Wilgotnosc", "Predkosc_Wiatru", "Zachmurzenie"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn do budowy datasetu LLM: {missing}")

    rows = []
    for _, row in df.iterrows():
        city = row["Miasto"]
        opis = row.get("Opis", "brak opisu")
        input_text = (
            f"wygeneruj raport pogody: miasto {city}; "
            f"temperatura {_num(row['Temperatura'])} C; "
            f"temperatura odczuwalna {_num(row['Temp_Odczuwalna'])} C; "
            f"cisnienie {int(row['Cisnienie'])} hPa; "
            f"wilgotnosc {int(row['Wilgotnosc'])} procent; "
            f"wiatr {_num(row['Predkosc_Wiatru'])} m/s; "
            f"zachmurzenie {int(row['Zachmurzenie'])} procent"
        )
        target_text = (
            f"W mieście {city} temperatura wynosi {_num(row['Temperatura'])} stopni C, "
            f"a temperatura odczuwalna {_num(row['Temp_Odczuwalna'])} stopni C. "
            f"Ciśnienie wynosi {int(row['Cisnienie'])} hPa, wilgotność {int(row['Wilgotnosc'])}%, "
            f"prędkość wiatru {_num(row['Predkosc_Wiatru'])} m/s, a zachmurzenie {int(row['Zachmurzenie'])}%. "
            f"Opis pogody: {opis}."
        )
        rows.append({"input_text": input_text, "target_text": target_text})

    dataset = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Zapisano dataset LLM: {output_csv} ({len(dataset)} przykładów)")
    return dataset


def compute_text_metrics(predictions, references) -> dict:
    """Liczy BLEU (sacrebleu) oraz ROUGE-1/2/L na parach predykcja-referencja."""
    import sacrebleu
    from rouge_score import rouge_scorer

    preds = [p.strip() for p in predictions]
    refs = [r.strip() for r in references]

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "bleu": round(float(bleu), 2),
        "rouge1": round(float(np.mean(r1)) * 100, 2),
        "rouge2": round(float(np.mean(r2)) * 100, 2),
        "rougeL": round(float(np.mean(rl)) * 100, 2),
    }


def train_model(dataset_csv: str, output_dir: str, epochs: int, batch_size: int, eval_path: str, lr: float = DEFAULT_LR) -> None:
    """Uruchamia fine-tuning PLT5-small i liczy BLEU/ROUGE na zbiorze ewaluacyjnym."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Brakuje bibliotek LLM. Zainstaluj zależności z requirements.txt.") from exc

    df = pd.read_csv(dataset_csv, encoding="utf-8-sig")
    if len(df) < 30:
        print("UWAGA: dataset jest bardzo mały, więc wyniki mogą być podatne na overfitting i halucynacje liczb.")
    print(f"Hiperparametry: epochs={epochs}, lr={lr}, batch_size={batch_size}")

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(batch["input_text"], max_length=128, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=GEN_MAX_LEN, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_text_metrics(decoded_preds, decoded_labels)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_LEN,
        generation_num_beams=4,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    device = model.device
    examples = []
    for _, row in eval_df.head(3).iterrows():
        enc = tokenizer(row["input_text"], return_tensors="pt", truncation=True, max_length=128).to(device)
        out = model.generate(**enc, max_length=GEN_MAX_LEN, num_beams=4, no_repeat_ngram_size=NO_REPEAT_NGRAM)
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        examples.append({
            "input_text": row["input_text"],
            "target_text": row["target_text"],
            "generated_text": gen,
        })

    report = {
        "model": MODEL_NAME,
        "num_examples_total": int(len(df)),
        "num_train": int(len(train_df)),
        "num_eval": int(len(eval_df)),
        "epochs": epochs,
        "learning_rate": lr,
        "metrics": {
            "eval_loss": round(float(metrics.get("eval_loss", float("nan"))), 4),
            "bleu": metrics.get("eval_bleu"),
            "rouge1": metrics.get("eval_rouge1"),
            "rouge2": metrics.get("eval_rouge2"),
            "rougeL": metrics.get("eval_rougeL"),
        },
        "examples": examples,
    }

    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Zapisano model: {output_dir}")
    print(f"Zapisano metryki i przykłady: {eval_path}")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


def generate_examples(model_dir: str = DEFAULT_MODEL_DIR, dataset_csv: str = DEFAULT_DATASET, n: int = 5) -> None:
    """Ładuje wytrenowany model i wypisuje przykładowe generacje (do raportu/prezentacji)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    df = pd.read_csv(dataset_csv, encoding="utf-8-sig").head(n)
    for _, row in df.iterrows():
        enc = tokenizer(row["input_text"], return_tensors="pt", truncation=True, max_length=128)
        out = model.generate(**enc, max_length=GEN_MAX_LEN, num_beams=4, no_repeat_ngram_size=NO_REPEAT_NGRAM)
        print("WEJŚCIE:   ", row["input_text"])
        print("MODEL:     ", tokenizer.decode(out[0], skip_special_tokens=True))
        print("OCZEKIWANE:", row["target_text"])
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true", help="wypisz przykłady generacji z zapisanego modelu")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    if args.generate:
        generate_examples(args.model_dir, args.dataset)
        return

    build_dataset(args.input, args.dataset)
    if args.train:
        train_model(args.dataset, args.model_dir, args.epochs, args.batch_size, args.eval_path, args.lr)
    elif not args.build_only:
        print("Dataset został przygotowany. Aby uruchomić fine-tuning, dodaj flagę --train.")


if __name__ == "__main__":
    main()