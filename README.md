# BH-Speech

Bosnian and Balkan Speech AI — otvoreni govorni AI, prvo ASR.
**Status: Faza 0 — istraživanje.** Bez treninga, bez redistribucije tuđeg audija.

- Kolekcija: https://huggingface.co/collections/BH-Language-AI/bh-speech
- Kontakt: xnetba@gmail.com

## Dokumenti
- `docs/PROJECT_CHARTER.md` — misija, pravila (test split se ne dira; licenca prije treninga)
- `docs/DATASET_AUDIT.md` — tehnički audit (splitovi, audio, duplikati, leakage=0)
- `docs/MODELS.md` — Whisper-large-v3 (Apache-2.0) vs BuzzASR (MIT) sažetak
- `docs/PLAN.md` — faze A–E, eval metrike, compute procjena

## Ključni nalazi Faze 0
1. Obje baze pravno čiste (Apache-2.0 / MIT); BuzzASR ~1.1x bolji CER na FLEURS-u.
2. Shunya dataset = prepakovani FLEURS (CC-BY-4.0) — koristiti original uz atribuciju.
3. Test (925) rezerviran; splitovi bez leakagea; ~14.5h govora.
4. AgentMujo Training nema Whisper podršku — treba speech adapter.
