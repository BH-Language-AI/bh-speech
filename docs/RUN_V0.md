# RUN v0 — prva produkcijska verzija (2026-11-26/27, LUMI)

Cilj: prvi objavljeni bosanski ASR (whisper-large-v3 FT na FLEURS `bs_ba`).
Rezultat: `BH-Language-AI/bh-speech-1-2b` — test **WER 0.1483 / CER 0.0531** (n=925, zamrznuto).

## Tok (sve preko LUMI workera, Contabo se ne dira za teški račun)

| Job / korak | Ishod |
|---|---|
| Kaggle `admiragic/bh-speech-train-lora-v0` | PAO — peft/torchao konflikt (CUDA wheel na Kaggle P100 stacku). Odbačeno, LUMI pobijedio kao kod grana-p. |
| LUMI 24044810 (rev1, datasets-lib) | PAO — `datasets>=5` traži CUDA `torchcodec` za Audio decode. |
| LUMI 24046062 (rev2, +torchcodec) | PAO — PyPI torchcodec je CUDA build (`libnvrtc.so.13`), ne radi na ROCm. |
| LUMI probe (forward izolacija) | Našao pravi uzrok LoRA pada (vidi ispod). Usput: soundfile ne čita neekstrahovane wavs — popravljeno. |
| LUMI 24046342 (rev2 LoRA trening) | PAO na step 0: peft 0.21.2 `PeftModelForSeq2SeqLM` injektira `input_ids=None` u `**kwargs` → sudar u `WhisperDecoder` na transformers 5.19.0. Verificirano čitanjem instaliranog koda na LUMI-ju. Kućni pinovi su za causal-LM, tamo radi. |
| **LUMI 24048537 (rev3, full FT)** | **GOTOVO** — final loss 1.7539; dev 0.1492/0.0552 (400); test 0.1483/0.0531 (925, jednom). |

## Zašto full FT, a ne LoRA

peft+transformers kombinacija je slomljena za Whisper na kućnim pinovima (5.19.0).
Full FT bf16 je ionako najbliže BuzzASR receptu ("simple fine-tuning").
Rev2 LoRA skripta je arhivirana (`slurm/train_bh_speech_lora.py`) za kasnije kad se peft popravi.

## Brojke (normalizer: BasicTextNormalizer)

- Baseline stock whisper-large-v3 (dev/100): WER 0.1814 / CER 0.0696
- v0 dev (400): WER 0.1492 / CER 0.0552
- v0 test (925, zamrznut): WER 0.1483 / CER 0.0531
- Referenca BuzzASR/bosnian (test): WER 0.1346 / CER 0.0359 — v0 je ispod, iskreno u kartici.

## Artefakti

- LUMI scratch: `/scratch/project_465003647/bh-speech/lora-v0/` (skripte + `out/metrics.txt`; teški `data/`, `ckpt/` očišćeni nakon objave)
- HF: `BH-Language-AI/bh-speech-1-2b` (model.safetensors 3.09 GB + kartica + logo)
- Stari `bh-speech-asr`: arhiva sa deprecated noticeom.
