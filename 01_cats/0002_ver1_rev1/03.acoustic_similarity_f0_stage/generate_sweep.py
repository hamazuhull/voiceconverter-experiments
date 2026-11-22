"""Subexperiment 03 — supplemental data generation: decoder pitch sweep.

Question: when the decoder fails to render a cat's F0 even though it is
injected (setting oracle_fo_vf in generate.py), is that because of the
absolute frequency, or because of the cat's conditioning (speaker code /
content features)?

Each clip from generate.py is re-synthesized with its own model condition
(z_q), its true voicing, and its true F0 shifted by k semitones:
  humans: k = 0, +6, +12, +18, +24   (pushes speech into the cat range)
  cats:   k = 0, -6, -12, -18, -24   (pulls cat calls into the human range)
If rendering accuracy depends on the injected frequency alone, the two groups
fall on one curve.

Outputs: data/sweep/<clip>__<k>.wav (requires data/ from generate.py).
"""
import csv
import pathlib

import numpy as np
import soundfile as sf

import model as stage_model
from generate import frames_to_samples

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'
SR = stage_model.SR
SHIFTS = {'cat': (0, -6, -12, -18, -24), 'human': (0, 6, 12, 18, 24)}


def main():
    (DATA / 'sweep').mkdir(parents=True, exist_ok=True)
    with open(DATA / 'manifest.csv') as f:
        manifest = list(csv.DictReader(f))
    m = stage_model.StageModel()
    for i, row in enumerate(manifest):
        kind = 'cat' if row['group'] == 'cat' else 'human'
        x, _ = sf.read(str(DATA / 'input' / f"{row['clip']}.wav"))
        n = x.size
        fr = np.load(DATA / 'frames' / f"{row['clip']}.npz")
        fo, vf = frames_to_samples(fr['f0_true'], fr['mask_true'], n)
        _, z_q, _ = m.speaker(stage_model.trim_silence(x))
        cond = m.condition(x, z_q)
        for k in SHIFTS[kind]:
            y = m.synth(m.replace(cond, n, fo=fo * 2 ** (k / 12), vf=vf), n)
            sf.write(str(DATA / 'sweep' / f"{row['clip']}__{k:+d}.wav"), y, SR, subtype='PCM_24')
        print(f"[{i + 1}/{len(manifest)}] {row['clip']} shifts {SHIFTS[kind]}", flush=True)


if __name__ == '__main__':
    main()
