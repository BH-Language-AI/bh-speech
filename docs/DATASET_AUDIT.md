# BH-Speech — Faza 0: dataset audit (tehnički)

Datum: 2026-11-11. Metoda: puni download (3.1GB) + lokalna analiza svih 4416 zapisa (pandas/soundfile). Test split diran SAMO za čitanje statistike, ne za trening.

## Splitovi: 3091 / 400 / 925 ✅ (odgovara FLEURS bs_ba)
## Audio: svi 16kHz, 0 korumpiranih/praznih
- train: 10.0h (3.6–35.8s, med 11.2s); val: 1.3h; test: 3.2h (5.5–37.6s). Ukupno ~14.5h.
## Transkripti
- Praznih: 0. Kolone: samo `audio` + `transcript` (metapodaci ogoljeni vs FLEURS).
- Duplikati unutar splita: train 1614/3091, val 250/400, test 576/925 — ista FLoRes rečenica, različiti govornici (očekivano za FLEURS, ali test ima ~349 jedinstvenih tekstova).
- **Leakage između splitova: 0** — splitovi čisti.
## Licenca (vidi shunya-license izvještaj)
Bez deklarisane licence; sadržaj = FLEURS CC-BY-4.0 → atribucija Google obavezna; koristiti original `google/fleurs:bs_ba`.
## Odluke
- Dev: train+val originalnog FLEURS-a; test (925) se ne otvara do zamrznute evaluacije.
- Prije treninga: normalizacijska pravila + manifesti (Faza 4).
