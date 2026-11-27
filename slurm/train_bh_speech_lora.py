#!/usr/bin/env python3
"""BH-Speech v0 (rev2) — whisper-large-v3 LoRA bf16 na LUMI (MI250X, ROCm).
BEZ datasets-lib Audio dekodiranja (datasets>=5 trazi CUDA torchcodec):
FLEURS se vuce kao sirovi fajlovi (tsv + wav.tar.gz), cita se soundfileom.
Splitovi: train 3091 / dev 400 (val) / test 925 (JEDNOM na kraju).
Izlaz: adapter + processor + metrics.txt u OUT.
Upotreba: python -u train_bh_speech.py --out /job/out
"""
import argparse, os, tarfile
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 3)))
ap.add_argument("--rank", type=int, default=int(os.environ.get("LORA_R", 32)))
ap.add_argument("--alpha", type=int, default=int(os.environ.get("LORA_ALPHA", 64)))
ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", 1e-4)))
ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 8)))
ap.add_argument("--grad-acc", type=int, default=int(os.environ.get("GRAD_ACC", 4)))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(f"{a.out}/data", exist_ok=True)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset as TorchDataset  # noqa: E402
assert torch.cuda.is_available(), "nema GPU-a"
print("cuda:", torch.cuda.get_device_name(0), torch.__version__, flush=True)

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,  # noqa: E402
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model  # noqa: E402
from jiwer import wer as jiwer_wer, cer as jiwer_cer  # noqa: E402
try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    norm = BasicTextNormalizer()
except Exception:
    norm = lambda s: s.lower().strip()  # noqa: E731

BASE = "openai/whisper-large-v3"
LANG, TASK = "bs", "transcribe"

# 1. SIROVI FLEURS bs_ba (bez datasets-lib): tsv + wav
print("skidam FLEURS bs_ba ...", flush=True)
repo = snapshot_download("google/fleurs", repo_type="dataset",
                         allow_patterns=["data/bs_ba/*"])
print("repo:", repo, flush=True)
items = {}
for split, tsv in (("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")):
    with tarfile.open(f"{repo}/data/bs_ba/audio/{split}.tar.gz") as t:
        t.extractall(f"{a.out}/data")
    rows = []
    for line in open(f"{repo}/data/bs_ba/{tsv}", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        assert len(parts) >= 3, parts[:2]
        wav = f"{a.out}/data/{split}/{parts[1]}"
        assert os.path.exists(wav), wav
        rows.append((wav, parts[2]))
    items[split] = rows
    print(split, "n =", len(rows), flush=True)
assert (len(items["train"]), len(items["dev"]), len(items["test"])) == (3091, 400, 925)
print("TEST ZAKLJUCAN: n = 925 (samo krajnja evaluacija)", flush=True)

proc = WhisperProcessor.from_pretrained(BASE, language=LANG, task=TASK)

def load_wav(path):
    arr, sr = sf.read(path, dtype="float32")
    assert sr == 16000, (path, sr)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr

class FleursBS(TorchDataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        path, text = self.rows[i]
        feats = proc.feature_extractor(load_wav(path), sampling_rate=16000).input_features[0]
        return {"input_features": feats, "labels": proc.tokenizer(text).input_ids}

train_ds = FleursBS(items["train"])
val_full = FleursBS(items["dev"])
val_ds = torch.utils.data.Subset(val_full, range(min(100, len(val_full))))
print("train:", len(train_ds), "val_eval:", len(val_ds), "val_full:", len(val_full), flush=True)

from dataclasses import dataclass  # noqa: E402

@dataclass
class Collator:
    processor: object
    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        lb = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features],
                                          return_tensors="pt")
        ids = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch

def decode_rows(model, rows):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for path, text in rows:
            feats = proc.feature_extractor(load_wav(path), sampling_rate=16000,
                                           return_tensors="pt").input_features.to(model.device).to(torch.bfloat16)
            ids = model.generate(feats, num_beams=1, no_repeat_ngram_size=3)
            preds.append(norm(proc.batch_decode(ids, skip_special_tokens=True)[0]))
            refs.append(norm(text))
    return {"wer": jiwer_wer(refs, preds), "cer": jiwer_cer(refs, preds), "n": len(rows)}

print("=== BASELINE (stock, dev n=100) ===", flush=True)
stock = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
stock.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language=LANG, task=TASK)
stock.config.suppress_tokens = []
bl = decode_rows(stock, items["dev"][:100])
print(f"BASELINE dev: WER={bl['wer']:.4f} CER={bl['cer']:.4f} (n={bl['n']})", flush=True)
del stock
torch.cuda.empty_cache()

print("=== LoRA TRENING ===", flush=True)
model = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
model.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language=LANG, task=TASK)
model.config.suppress_tokens = []
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                        target_modules=["q_proj", "v_proj"],
                                        task_type="SEQ_2_SEQ_LM"))
model.print_trainable_parameters()

def compute_metrics(pred):
    ids, refs = pred.predictions, pred.label_ids
    refs[refs == -100] = proc.tokenizer.pad_token_id
    p = [norm(s) for s in proc.batch_decode(ids, skip_special_tokens=True)]
    r = [norm(s) for s in proc.batch_decode(refs, skip_special_tokens=True)]
    return {"wer": jiwer_wer(r, p), "cer": jiwer_cer(r, p)}

args = Seq2SeqTrainingArguments(
    output_dir=f"{a.out}/ckpt", per_device_train_batch_size=a.batch,
    per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.grad_acc,
    learning_rate=a.lr, warmup_steps=200, num_train_epochs=a.epochs,
    bf16=True, gradient_checkpointing=True, predict_with_generate=True,
    generation_num_beams=1, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
    logging_steps=25, save_total_limit=1, remove_unused_columns=False,
    label_names=["labels"], report_to="none")
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds,
                         eval_dataset=val_ds, data_collator=Collator(proc),
                         compute_metrics=compute_metrics, processing_class=proc)
res = trainer.train()
print("TRENING GOTOV — final loss:", res.training_loss, flush=True)

print("=== PUNI DEV (n=400) ===", flush=True)
vf = decode_rows(trainer.model, items["dev"])
print(f"DEV: WER={vf['wer']:.4f} CER={vf['cer']:.4f} (n={vf['n']})", flush=True)
print("=== ZAMRZNUTI TEST (jednom) ===", flush=True)
te = decode_rows(trainer.model, items["test"])
print(f"TEST: WER={te['wer']:.4f} CER={te['cer']:.4f} (n={te['n']})", flush=True)

trainer.save_model(f"{a.out}/adapter")
proc.save_pretrained(f"{a.out}/adapter")
with open(f"{a.out}/metrics.txt", "w") as f:
    f.write(f"baseline_dev100: WER={bl['wer']:.4f} CER={bl['cer']:.4f}\n")
    f.write(f"dev: WER={vf['wer']:.4f} CER={vf['cer']:.4f} (n={vf['n']})\n")
    f.write(f"test: WER={te['wer']:.4f} CER={te['cer']:.4f} (n={te['n']})\n")
print("SACUVANO:", a.out, flush=True)
