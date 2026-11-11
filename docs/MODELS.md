# Whisper-large-v3 vs BuzzASR/bosnian (BH-Speech Faza 0)

Datum: 2026-11-11. Izvori: HF model cardovi, config.json, HF API.

## openai/whisper-large-v3
- Encoder-decoder Transformer, 32+32 sloja, d_model 1280, 20 glava, 1.543B F16
- 99 jezika (bs token `<|bs|>:50315`); 16kHz, 30s chunk, 128 Mel binova
- Tokenizer: byte-BPE, vocab 51866
- **Licenca: Apache-2.0** (komercijala + redistribucija DA, uz attribution/NOTICE/state-changes)
- Revizija: `06f233fe06e710322aca913c1bc4249a0d71fce1` (2024-08-12)
- Bosanski WER na kartici: NEPROVJERENO (samo opšti prosjeci)
- Ograničenja: halucinacije, slabiji low-resource jezici, repetitivni ispisi, nije real-time, zabrana nadzora/klasifikacije atributa

## BuzzASR/bosnian
- Održava: BuzzASR/Lemn Lab (+EleutherAI); autori Singh, Yadavalli, Arnett, Warstadt; EMNLP 2026, arXiv:2609.09554
- Baza: whisper-large-v3 (ista arh); fine-tune, `language:bs` zapečen u generation_config
- **Licenca: MIT deklarisano** (nema LICENSE fajla u repou — NEPROVJERENO); lanac Apache-2.0→MIT kompatibilan
- Eval (normalizovano, FLEURS): CER 3.59 / WER 13.46 vs baza zero-shot CER 3.87 (~1.1x)
- Trening: FLEURS + Goldfish text-only
- Revizija: `a02491a35efbad389ee478aac0b6b81d174b651e` (2026-09-22)
- Ograničenja: samo bosanski; evaluirano samo FLEURS/CV — ostali domeni nepoznati

## Zaključak
Oba pravno čista za komercijalu (uz atribucije). BuzzASR ~1.1x bolji CER na FLEURS-u, ali su brojke samo FLEURS — prije bilo kakve tvrdnje treba vlastiti benchmark na bs domenu (Faza 5 plana).
