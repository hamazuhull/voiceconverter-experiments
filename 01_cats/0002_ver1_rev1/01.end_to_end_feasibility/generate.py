"""Subexperiment 01 — data generation.

Selects human source utterances (Common Voice ja), cat references (meow-10k)
and human reference speakers (Common Voice ja = open set, jvs = closed set),
then converts every source to every reference with the pretrained NVCNSFV2 model. For cat references the decoder-side
`semitone_shift` is set per pair so that the output median F0 matches the cat
reference median F0 (pYIN); the unshifted output is kept in data/conv_s0/.

Outputs (inside this directory):
  data/src/*.wav          source utterances (24 kHz)
  data/ref/*.wav          silence-trimmed references (24 kHz)
  data/conv/*.wav         converted utterances (24 kHz); cat refs: pitch-matched
  data/conv_s0/*.wav      cat refs only: converted with semitone_shift = 0
  data/conv_fo/*.npy      model-predicted F0 (Hz, per sample) of each conversion
  data/cat_candidates.csv voicing screening of cat clips
  data/manifest.csv       one row per conversion
"""
import argparse
import csv
import pathlib
import random

import numpy as np
import librosa
import soundfile as sf

import vc

HERE = pathlib.Path(__file__).resolve().parent
CV_DIR = pathlib.Path('/path/to/datasets/common-voice-scripted-speech-27-0-japane-ea5ad4e9/cv-corpus-27.0-2026-09-11/ja')
MEOW_DIR = pathlib.Path('/path/to/datasets/meow-10k/audios')
JVS_DIR = pathlib.Path('/path/to/datasets/jvs_ver1')
GENDER = {'male_masculine': 'M', 'female_feminine': 'F'}
MIN_REF_SEC = 4.0
MEOW_F0_RANGE = (250.0, 1500.0)
MAX_SHIFT = 24.0  # semitones, safety clip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-src-per-gender', type=int, default=4)
    p.add_argument('--n-ref-per-gender', type=int, default=2)
    p.add_argument('--n-cat', type=int, default=6)
    p.add_argument('--n-cat-pool', type=int, default=120)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=pathlib.Path, default=HERE / 'data')
    return p.parse_args()


def select_cv(rng, n_src, n_ref):
    """Pick distinct CV speakers: one source utterance each, and reference speakers."""
    durations = {}
    with open(CV_DIR / 'clip_durations.tsv') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            durations[row['clip']] = float(row['duration[ms]']) / 1000
    by_spk = {}
    with open(CV_DIR / 'test.tsv') as f:
        for row in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if row['gender'] not in GENDER:
                continue
            by_spk.setdefault((row['client_id'], GENDER[row['gender']]), []).append(row['path'])
    sources, refs = [], []
    for g in ('M', 'F'):
        spks = sorted(k for k, clips in by_spk.items() if k[1] == g and len(clips) >= 3)
        rng.shuffle(spks)
        n_g = 0
        for spk in list(spks):
            clips = [c for c in sorted(by_spk[spk]) if 3.0 <= durations.get(c, 0) <= 7.0]
            if clips:
                sources.append({'id': f'cv{g}{n_g:02d}', 'gender': g, 'speaker': spk[0][:12],
                                'files': [str(CV_DIR / 'clips' / rng.choice(clips))]})
                spks.remove(spk)
                n_g += 1
                if n_g == n_src:
                    break
        for k, spk in enumerate(spks[:n_ref]):  # disjoint from the source speakers
            refs.append({'id': f'cvref{g}{k:02d}', 'kind': 'human_open', 'gender': g,
                         'speaker': spk[0][:12],
                         'files': [str(CV_DIR / 'clips' / c) for c in sorted(by_spk[spk])[:6]]})
    return sources, refs


def select_jvs(n_ref):
    spk_gender = {}
    with open(JVS_DIR / 'gender_f0range.txt') as f:
        next(f)
        for line in f:
            spk, g = line.split()[:2]
            spk_gender[spk] = g
    refs = []
    for g in ('M', 'F'):
        for spk in sorted(s for s, sg in spk_gender.items() if sg == g)[:n_ref]:
            wavs = sorted((JVS_DIR / spk / 'nonpara30' / 'wav24kHz16bit').glob('*.wav'))[:3]
            refs.append({'id': f'jvs{spk[3:]}{g}', 'kind': 'human_closed', 'gender': g,
                         'speaker': spk, 'files': [str(w) for w in wavs]})
    return refs


def screen_cats(rng, n_pool, n_cat, out_csv):
    """Pick cat clips that contain meow-like tonal vocalization.

    Ranked by pYIN voiced ratio (150-2000 Hz); clips whose voiced median F0 is
    outside MEOW_F0_RANGE (purrs, hums, low-frequency noise pinned near fmin)
    are not eligible.
    """
    files = sorted(MEOW_DIR.glob('*.wav'))
    pool = rng.sample(files, n_pool)
    rows = []
    for i, path in enumerate(pool):
        x = vc.trim_silence(vc.load_audio(path, sr=16000), sr=16000)
        f0, voiced, _ = librosa.pyin(x, fmin=150, fmax=2000, sr=16000, frame_length=1024)
        voiced_ratio = float(np.mean(voiced))
        med = float(np.nanmedian(f0[voiced])) if voiced.any() else float('nan')
        rows.append({'file': path.name, 'trimmed_sec': round(x.size / 16000, 2),
                     'voiced_ratio': round(voiced_ratio, 4), 'median_f0_hz': round(med, 1)})
        print(f'[screen {i + 1}/{n_pool}] {path.name} voiced={voiced_ratio:.3f} f0={med:.0f}', flush=True)
    rows.sort(key=lambda r: -r['voiced_ratio'])
    lo, hi = MEOW_F0_RANGE
    eligible = [r for r in rows if lo <= r['median_f0_hz'] <= hi]  # NaN compares False
    selected = eligible[:n_cat]
    for r in rows:
        r['selected'] = int(r in selected)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return [{'id': f'cat{k:02d}', 'kind': 'cat', 'gender': '-', 'speaker': r['file'].removesuffix('.wav'),
             'files': [str(MEOW_DIR / r['file'])]} for k, r in enumerate(selected)]


def build_reference(files):
    """Concatenate silence-trimmed files until at least MIN_REF_SEC seconds are collected."""
    parts = []
    for path in files:
        parts.append(vc.trim_silence(vc.load_audio(path)))
        if sum(p.size for p in parts) >= MIN_REF_SEC * vc.SR:
            break
    return np.concatenate(parts)


def median_f0(x24k):
    """Median voiced pYIN F0 (Hz) of a 24 kHz waveform; same settings as cat screening."""
    x = librosa.resample(np.asarray(x24k, dtype=np.float64), orig_sr=vc.SR, target_sr=16000)
    f0, voiced, _ = librosa.pyin(x, fmin=150, fmax=2000, sr=16000, frame_length=1024)
    return float(np.nanmedian(f0[voiced])) if voiced.any() else float('nan')


def condition_of(src, ref):
    if ref['kind'] == 'cat':
        return 'human2cat'
    scope = 'open' if ref['kind'] == 'human_open' else 'closed'
    return f"{scope}_{'intra' if src['gender'] == ref['gender'] else 'inter'}"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    for sub in ('src', 'ref', 'conv', 'conv_s0', 'conv_fo'):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    sources, cv_refs = select_cv(rng, args.n_src_per_gender, args.n_ref_per_gender)
    refs = cv_refs + select_jvs(args.n_ref_per_gender) + screen_cats(
        rng, args.n_cat_pool, args.n_cat, args.out / 'cat_candidates.csv')

    model = vc.VoiceConverter()
    print(f'loaded checkpoint step {model.step}', flush=True)

    for ref in refs:
        r = build_reference(ref['files'])
        sf.write(str(args.out / 'ref' / f"{ref['id']}.wav"), r, vc.SR, subtype='PCM_24')
        ref['z_q'] = model.speaker_encode(r)['z_q']
        if ref['kind'] == 'cat':
            ref['f0_hz'] = median_f0(r)
    for src in sources:
        x = vc.load_audio(src['files'][0])
        sf.write(str(args.out / 'src' / f"{src['id']}.wav"), x, vc.SR, subtype='PCM_24')
        src['x'] = x
        src['z_q'] = model.speaker_encode(x)['z_q']

    rows = []
    for src in sources:
        for ref in refs:
            name = f"{src['id']}__{ref['id']}"
            shift, target_f0 = 0.0, float('nan')
            y, fo, _ = model.convert(src['x'], ref['z_q'], z_src=src['z_q'])
            if ref['kind'] == 'cat':
                sf.write(str(args.out / 'conv_s0' / f'{name}.wav'), y, vc.SR, subtype='PCM_24')
                target_f0, f0_s0 = ref['f0_hz'], median_f0(y)
                if np.isfinite(f0_s0) and np.isfinite(target_f0):
                    shift = float(np.clip(12 * np.log2(target_f0 / f0_s0), -MAX_SHIFT, MAX_SHIFT))
                y, fo, _ = model.convert(src['x'], ref['z_q'], z_src=src['z_q'], semitone_shift=shift)
            if not np.all(np.isfinite(y)):
                raise RuntimeError(f'non-finite output for {name}')
            sf.write(str(args.out / 'conv' / f'{name}.wav'), y, vc.SR, subtype='PCM_24')
            np.save(args.out / 'conv_fo' / f'{name}.npy', fo[::240])  # 100 frames/s
            rows.append({
                'conv': f'conv/{name}.wav', 'src': f"src/{src['id']}.wav", 'ref': f"ref/{ref['id']}.wav",
                'src_id': src['id'], 'src_gender': src['gender'], 'src_file': pathlib.Path(src['files'][0]).name,
                'ref_id': ref['id'], 'ref_kind': ref['kind'], 'ref_gender': ref['gender'],
                'ref_speaker': ref['speaker'], 'condition': condition_of(src, ref),
                'model_step': model.step, 'shift_st': round(shift, 3), 'target_f0_hz': round(target_f0, 1),
            })
            print(f'[convert] {name} ({rows[-1]["condition"]}, shift {shift:+.1f} st)', flush=True)

    with open(args.out / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(sources)} sources x {len(refs)} references = {len(rows)} conversions')


if __name__ == '__main__':
    main()
