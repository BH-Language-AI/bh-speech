# Shunya dataset — licenca i porijeklo (BH-Speech Faza 0)

Datum: 2026-11-11. Izvori: HF API, dataset README, shunyalabs.ai/terms, FLEURS card.

## Nalaz
- Deklarisana licenca: NE POSTOJI (API bez license polja; README prazan, samo frontmatter).
- Autor: Shunya Labs / Shunya Research Labs Pvt Ltd (Gurgaon, Indija); kontakt: grievance officer + `0@shunyalabs.ai` (NEPROVJERENO).
- Porijeklo: prepakovani FLEURS `bs_ba` — dokazi: 4416 = 3091+400+925, isti splitovi, 16kHz, FLoRes Wikipedija rečenice, `transcription`→`transcript`, odbačeni metapodaci; org ima 81 dataset = 81 FLEURS jezik.
- FLEURS licenca: CC-BY-4.0 (Google, arXiv 2205.12446).

## Odluka
Dataset se tretira kao CC-BY-4.0: treniranje/komercijala/redistribucija DA — uz obaveznu atribuciju Google FLEURS. Najsigurnije: koristiti original `google/fleurs` config `bs_ba`. Ne predstavljati kao vlasnički Shunya dataset. Test split (925) rezerviran, ne dirati.
