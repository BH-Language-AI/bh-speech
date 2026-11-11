"""Validacija govornog dataseta: splitovi, audio, duplikati, leakage.

Koristenje:
    python scripts/validate_dataset.py /put/do/dataseta --reserved-test
Test split se samo cita za statistiku — nikad za trening.
"""
import argparse
import glob
import io
import statistics
import sys

import pandas as pd


def load_split(root, split):
    files = sorted(glob.glob(f'{root}/data/{split}-*.parquet'))
    if not files:
        raise SystemExit(f'nema fajlova za split {split} u {root}/data')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def audio_stats(df):
    import soundfile as sf
    bad, srs, durs = 0, set(), []
    for b in df['audio']:
        try:
            d, sr = sf.read(io.BytesIO(b['bytes']))
            srs.add(sr)
            durs.append(len(d) / sr)
            if (abs(d) < 1e-6).all():
                bad += 1
        except Exception:
            bad += 1
    return {'sr': sorted(srs), 'n': len(durs),
            'min': min(durs), 'max': max(durs),
            'med': statistics.median(durs), 'sati': sum(durs) / 3600,
            'losih': bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='dir dataseta (sa data/*.parquet)')
    args = ap.parse_args()
    ok = True
    tekstovi = {}
    for split in ['train', 'validation', 'test']:
        df = load_split(args.root, split)
        t = df['transcript']
        tekstovi[split] = set(t)
        st = audio_stats(df)
        print(f'{split}: n={len(df)} praznih={int((t.str.len() == 0).sum())} '
              f'dup={int(t.duplicated().sum())} sr={st["sr"]} '
              f'sati={st["sati"]:.1f} losih={st["losih"]}')
        if st['losih'] or set(st['sr']) != {16000}:
            ok = False
    tr, va, te = tekstovi['train'], tekstovi['validation'], tekstovi['test']
    leak = len(tr & te) + len(tr & va) + len(va & te)
    print(f'leakage izmedju splitova: {leak}')
    if leak:
        ok = False
    print('REZULTAT:', 'OK' if ok else 'NEISPRAVNO')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
