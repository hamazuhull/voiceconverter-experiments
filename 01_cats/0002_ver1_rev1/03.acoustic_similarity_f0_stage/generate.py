"""Subexperiment 03 — data generation: stage-wise F0 probes.

Every clip is reconstructed with its own speaker code (self-reconstruction),
so the input's pYIN F0 is frame-aligned ground truth for both stages:

  setting        speaker code   F0 / voicing fed to the decoder
  zq             z_q (as used)  model (normal operation)
  z              continuous z   model (bypasses the 512-code quantizer)
  oracle_fo      z_q            F0 := true F0 of the input; voicing from model
  oracle_fo_vf   z_q            F0 and voicing := true F0 / voicing of the input

Clips: meow-10k cats (screened as in subexp 01), Common Voice ja (open-set
humans, the subexp-01 sources) and jvs (closed-set humans, training speakers).

Outputs (inside this directory):
  data/input/<clip>.wav              input clip (24 kHz)
  data/output/<clip>__<setting>.wav  reconstructions
  data/frames/<clip>.npz             10 ms frame-level truth + condition parameters
  data/cat_candidates.csv            cat screening
  data/manifest.csv                  one row per clip
"""
import argparse
import csv
import importlib.util
import pathlib
import random

import numpy as np
import librosa
import soundfile as sf

import model as stage_model

HERE = pathlib.Path(__file__).resolve().parent
JVS_DIR = pathlib.Path('/path/to/datasets/jvs_ver1')
SR = stage_model.SR
HOP = 240                       # 10 ms frames
PYIN = dict(fmin=50.0, fmax=2000.0, sr=SR, frame_length=2048, hop_length=HOP)
CAT_MIN_F0 = 200.0              # cat frames below this are background, not the call (see subexp 01)
MAX_SEC = 8.0
SETTINGS = ('zq', 'z', 'oracle_fo', 'oracle_fo_vf')


def load_subexp01_generate():
    spec = importlib.util.spec_from_file_location(
        'gen01', HERE.parent / '01.end_to_end_feasibility' / 'generate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-cat', type=int, default=16)
    p.add_argument('--n-cat-pool', type=int, default=300)
    p.add_argument('--n-human-per-gender', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=pathlib.Path, default=HERE / 'data')
    return p.parse_args()


def select_jvs(n_per_gender):
    """Closed-set humans: jvs speakers not used as references in subexp 01, one parallel100 utterance each."""
    genders = {}
    with open(JVS_DIR / 'gender_f0range.txt') as f:
        next(f)
        for line in f:
            spk, g = line.split()[:2]
            genders[spk] = g
    clips = []
    for g in ('M', 'F'):
        spks = [s for s in sorted(genders) if genders[s] == g and s not in ('jvs001', 'jvs002', 'jvs003', 'jvs004')]
        for k, spk in enumerate(spks[:n_per_gender]):
            clips.append({'id': f'jvs{g}{k:02d}', 'group': 'human_closed', 'gender': g, 'speaker': spk,
                          'file': str(JVS_DIR / spk / 'parallel100' / 'wav24kHz16bit' / 'VOICEACTRESS100_001.wav')})
    return clips


def truth_f0(x, is_cat):
    """pYIN F0 (Hz, NaN when unvoiced) and the mask of frames trusted as ground truth."""
    f0, voiced, prob = librosa.pyin(x, **PYIN)
    mask = voiced & (prob > 0.5) & np.isfinite(f0)
    if is_cat:
        mask &= np.nan_to_num(f0) >= CAT_MIN_F0
    return f0, voiced, prob, mask


def frames_to_samples(f0, mask, n):
    """Per-sample oracle F0 (log-interpolated across untrusted frames) and voicing flag."""
    t_frame = np.arange(f0.size) * HOP
    t = np.arange(n)
    if mask.any():
        lf = np.interp(t, t_frame[mask], np.log(f0[mask]))
        fo = np.exp(lf)
    else:
        fo = np.full(n, 100.0)
    vf = np.interp(t, t_frame, mask.astype(float)) > 0.5
    return fo.astype(np.float32), vf.astype(np.float32)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    for sub in ('input', 'output', 'frames'):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    gen01 = load_subexp01_generate()
    # same seed / call order as subexp 01 -> the same 8 Common Voice sources
    cv_sources, _ = gen01.select_cv(rng, args.n_human_per_gender, 0)
    clips = [{'id': s['id'], 'group': 'human_open', 'gender': s['gender'], 'speaker': s['speaker'],
              'file': s['files'][0]} for s in cv_sources]
    clips += select_jvs(args.n_human_per_gender)
    cats = gen01.screen_cats(rng, args.n_cat_pool, args.n_cat, args.out / 'cat_candidates.csv')
    clips += [{'id': c['id'], 'group': 'cat', 'gender': '-', 'speaker': c['speaker'], 'file': c['files'][0]}
              for c in cats]

    m = stage_model.StageModel()
    print(f'loaded checkpoint step {m.step}', flush=True)

    rows = []
    for k, clip in enumerate(clips):
        is_cat = clip['group'] == 'cat'
        x = stage_model.load_audio(clip['file'])
        if is_cat:
            x = stage_model.trim_silence(x)
        x = x[:int(MAX_SEC * SR)]
        n = x.size
        sf.write(str(args.out / 'input' / f"{clip['id']}.wav"), x, SR, subtype='PCM_24')

        f0, voiced, prob, mask = truth_f0(x, is_cat)
        oracle_fo, oracle_vf = frames_to_samples(f0, mask, n)

        z, z_q, code = m.speaker(stage_model.trim_silence(x))
        cond_zq = m.condition(x, z_q)
        cond_z = m.condition(x, z)
        conds = {
            'zq': cond_zq,
            'z': cond_z,
            'oracle_fo': m.replace(cond_zq, n, fo=oracle_fo),
            'oracle_fo_vf': m.replace(cond_zq, n, fo=oracle_fo, vf=oracle_vf),
        }
        for setting in SETTINGS:
            y = m.synth(conds[setting], n)
            if not np.all(np.isfinite(y)):
                raise RuntimeError(f"non-finite output: {clip['id']} {setting}")
            sf.write(str(args.out / 'output' / f"{clip['id']}__{setting}.wav"), y, SR, subtype='PCM_24')

        n_frames = f0.size
        idx = np.minimum(np.arange(n_frames) * HOP, n - 1)
        save = {'f0_true': f0, 'voiced_true': voiced, 'prob_true': prob, 'mask_true': mask}
        for name, cond in (('zq', cond_zq), ('z', cond_z)):
            nc = m.numpy_cond(cond, n)
            for key, v in nc.items():
                save[f'{name}_{key}'] = v[idx]
        np.savez_compressed(args.out / 'frames' / f"{clip['id']}.npz", **save)

        z_np, zq_np = np.asarray(z).reshape(-1), np.asarray(z_q).reshape(-1)
        rows.append({
            'clip': clip['id'], 'group': clip['group'], 'gender': clip['gender'], 'speaker': clip['speaker'],
            'file': pathlib.Path(clip['file']).name, 'duration_sec': round(n / SR, 3), 'code': code,
            'z_to_zq_dist': round(float(np.linalg.norm(z_np - zq_np)), 4),
            'z_norm': round(float(np.linalg.norm(z_np)), 4),
            'truth_frames': int(mask.sum()), 'model_step': m.step,
        })
        print(f"[{k + 1}/{len(clips)}] {clip['id']} ({clip['group']}) code={code} truth_frames={mask.sum()}",
              flush=True)

    with open(args.out / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} clips x {len(SETTINGS)} settings')


if __name__ == '__main__':
    main()
