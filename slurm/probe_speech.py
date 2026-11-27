#!/usr/bin/env python3
"""Speech probe: plain forward whisper-large-v3+LoRA na LUMI (bez Trainera).
Odgovara: da li model(**batch) radi na transformers 5.19.0 + peft 0.21.2?
Upotreba: python -u probe_speech.py --out /job/out
"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
assert torch.cuda.is_available(), "nema GPU-a"
print("cuda:", torch.cuda.get_device_name(0), torch.__version__, flush=True)
import transformers, peft  # noqa: E402
print("transformers:", transformers.__version__, "peft:", peft.__version__, flush=True)

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

BASE = "openai/whisper-large-v3"
repo = snapshot_download("google/fleurs", repo_type="dataset",
                         allow_patterns=["data/bs_ba/train.tsv", "data/bs_ba/audio/train.tar.gz"])
import tarfile
tgz = f"{repo}/data/bs_ba/audio/train.tar.gz"
d = f"{a.out}/pdata"
os.makedirs(d, exist_ok=True)
rows = []
for line in open(f"{repo}/data/bs_ba/train.tsv", encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    rows.append((parts[1], parts[2]))
    if len(rows) == 2:
        break
with tarfile.open(tgz) as t:
    members = [t.getmember(f"train/{wav}") for wav, _ in rows]
    t.extractall(d, members=members)
print("extract:", [m.name for m in members], flush=True)

proc = WhisperProcessor.from_pretrained(BASE, language="bs", task="transcribe")
feats, labs = [], []
for wav, text in rows:
    arr, sr = sf.read(f"{d}/train/{wav}", dtype="float32")
    assert sr == 16000
    feats.append(proc.feature_extractor(arr, sampling_rate=16000).input_features[0])
    labs.append(proc.tokenizer(text).input_ids)
batch_in = proc.feature_extractor.pad([{"input_features": f} for f in feats], return_tensors="pt")
lb = proc.tokenizer.pad([{"input_ids": l} for l in labs], return_tensors="pt")
ids = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
batch_in["labels"] = ids[:, 1:] if (ids[:, 0] == proc.tokenizer.bos_token_id).all().item() else ids
print("batch keys:", list(batch_in.keys()),
      {k: tuple(v.shape) for k, v in batch_in.items()}, flush=True)

model = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
model.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language="bs", task="transcribe")
model.config.suppress_tokens = []
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                        target_modules=["q_proj", "v_proj"],
                                        task_type="SEQ_2_SEQ_LM"))
model.train()
batch_in = {k: v.to(model.device) for k, v in batch_in.items()}
try:
    out = model(**batch_in)
    print("FORWARD OK, loss =", float(out.loss), flush=True)
except TypeError as e:
    print("FORWARD FAIL:", str(e)[:1500], flush=True)
print("PROBE_DONE", flush=True)
