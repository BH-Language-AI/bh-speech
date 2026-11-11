# BH-Speech — Plan (Faza 0 zaključak)

## Gotovo
- Whisper/BuzzASR audit (licence, revizije, brojke) ✅
- Shunya licenca: CC-BY-4.0 tretman, atribucija obavezna ✅
- Dataset audit: splitovi, audio, duplikati, leakage=0 ✅
- Framework: whisper podrška NE POSTOJI (grep prazan) → speech adapter potreban

## Sljedeće (treba odobrenje/odluku)
1. Odluka: original `google/fleurs:bs_ba` kao kanonski dev izvor (preporuka DA)
2. Baseline eval Whisper vs BuzzASR na val (mali CPU/GPU smoke — dozvoljeno kad se definira)
3. Speech adapter u AgentMujo Training (nakon migracije frameworka)
4. LUMI compute plan za LoRA/full FT (procjena prije zahtjeva)
