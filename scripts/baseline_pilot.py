"""Pilot baseline: transkribuj N uzoraka iz validationa sa oba modela, izracunaj WER/CER.

Koristenje:
    python scripts/baseline_pilot.py --n 3 --out benchmarks/pilot_v1.json
Radi na CPU (sporo) ili GPU. Ne dira test split.
"""
import argparse
import glob
import io
import json
import time

import pandas as pd
import soundfile as sf


def load_samples(root, n):
    files = sorted(glob.glob(f'{root}/data/validation-*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--data', default='/tmp/opencode/shunya')
    ap.add_argument('--out', default='benchmarks/pilot_v1.json')
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from bh_speech.evaluation import cer, wer

    df = load_samples(args.data, args.n)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results = []
    for model_id in ['openai/whisper-large-v3', 'BuzzASR/bosnian']:
        local = '/tmp/opencode/models/' + model_id.split('/')[-1].replace('-', '_')
        t0 = time.time()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        proc = AutoProcessor.from_pretrained(local)
        pipe = pipeline('automatic-speech-recognition', model=model,
                        tokenizer=proc.tokenizer,
                        feature_extractor=proc.feature_extractor)
        for i, row in df.iterrows():
            data, sr = sf.read(io.BytesIO(row['audio']['bytes']))
            hyp = pipe({'array': data, 'sampling_rate': sr},
                       generate_kwargs={'language': 'bosnian', 'task': 'transcribe'})['text']
            ref = row['transcript']
            results.append({'model': model_id, 'i': int(i),
                            'wer': wer(hyp, ref), 'cer': cer(hyp, ref),
                            'hyp': hyp[:200], 'ref': ref[:200]})
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=1, ensure_ascii=False)
        dt = time.time() - t0
        print(f'{model_id}: gotovo za {dt / 60:.1f} min', flush=True)
        del model, proc, pipe
        import gc
        gc.collect()
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    for r in results:
        print(f"{r['model'].split('/')[-1]} [{r['i']}] WER={r['wer']:.3f} CER={r['cer']:.3f}")
    print('SPREMLJENO', args.out)


if __name__ == '__main__':
    main()
