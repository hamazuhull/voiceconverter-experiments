"""Subexperiment 01 — analysis.

Reads data/manifest.csv written by generate.py and measures, for every
conversion, how far the output is from its source and from its target
reference. Human-to-cat is compared with conventional VC baselines
(open/closed-set x intra/inter-gender).

Metrics (all audio resampled to 16 kHz so the 8 kHz-band-limited cat
recordings and 24 kHz human recordings share one bandwidth):
  mfcc_dist   Euclidean distance of utterance-mean MFCC c1..c24 over
              non-silent frames (librosa MFCC units, i.e. DCT of dB mel).
  ltas_dist   RMS difference (dB) of level-normalized long-term average
              spectra, 100 Hz - 7 kHz.
  spk_cos     cosine similarity of the model's continuous speaker
              embedding `z` (model-internal measure; reported, not used
              for gap_closed because it is too noisy).
  f0          pYIN median F0 of voiced frames, 60-1600 Hz search range.
  f0_meow     same with a 150-2000 Hz search range; used as the target F0
              of cat references, whose wide-range estimate can lock onto
              low-frequency background noise.
For each distance d: gap_closed = 1 - d(conv, ref) / d(src, ref)
(1 = reached the target, 0 = no movement, <0 = moved away).

Outputs: results/metrics.csv, results/summary.csv, results/summary.json,
         results/reference_spread.csv, results/*.png
"""
import argparse
import csv
import itertools
import json
import pathlib

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vc

HERE = pathlib.Path(__file__).resolve().parent
FEAT_SR = 16000
CONDITIONS = ['human2cat', 'open_intra', 'open_inter', 'closed_intra', 'closed_inter']
LABELS = {'human2cat': 'human→cat', 'open_intra': 'open intra-gender', 'open_inter': 'open inter-gender',
          'closed_intra': 'closed intra-gender', 'closed_inter': 'closed inter-gender'}
# Reference palette (dataviz skill), first three slots: validated all-pairs.
GROUP_COLOR = {'human2cat': '#eb6834', 'open': '#2a78d6', 'closed': '#1baf7a'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def color_of(cond):
    return GROUP_COLOR['human2cat' if cond == 'human2cat' else cond.split('_')[0]]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=pathlib.Path, default=HERE / 'data')
    p.add_argument('--out', type=pathlib.Path, default=HERE / 'results')
    return p.parse_args()


class Features:
    """Per-file features, cached by path."""

    def __init__(self, data_dir, model):
        self.data_dir, self.model, self.cache = data_dir, model, {}

    def __call__(self, rel):
        if rel not in self.cache:
            self.cache[rel] = self._compute(self.data_dir / rel)
        return self.cache[rel]

    def _compute(self, path):
        x24 = vc.load_audio(path)
        x = librosa.resample(x24, orig_sr=vc.SR, target_sr=FEAT_SR)
        # non-silent frames (40 dB below peak frame)
        rms = librosa.feature.rms(y=x, frame_length=512, hop_length=160)[0]
        active = librosa.amplitude_to_db(rms, ref=np.max) > -40
        mfcc = librosa.feature.mfcc(y=x, sr=FEAT_SR, n_mfcc=25, n_fft=512, hop_length=160,
                                    n_mels=40, fmin=0, fmax=8000)
        n = min(mfcc.shape[1], active.size)
        mfcc_mean = mfcc[1:, :n][:, active[:n]].mean(axis=1)
        S = np.abs(librosa.stft(x, n_fft=1024, hop_length=160)) ** 2
        S = S[:, :min(S.shape[1], active.size)][:, active[:min(S.shape[1], active.size)]]
        freqs = librosa.fft_frequencies(sr=FEAT_SR, n_fft=1024)
        band = (freqs >= 100) & (freqs <= 7000)
        ltas = 10 * np.log10(S.mean(axis=1)[band] + 1e-12)
        ltas -= ltas.mean()
        f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=1600, sr=FEAT_SR, frame_length=1024, hop_length=160)
        f0_med = float(np.nanmedian(f0[voiced])) if voiced.any() else float('nan')
        f0m, voicedm, _ = librosa.pyin(x, fmin=150, fmax=2000, sr=FEAT_SR, frame_length=1024, hop_length=160)
        f0m_med = float(np.nanmedian(f0m[voicedm])) if voicedm.any() else float('nan')
        z = np.asarray(self.model.speaker_encode(vc.trim_silence(x24))['z']).reshape(-1)
        return {'mfcc': mfcc_mean, 'ltas': ltas, 'ltas_freqs': freqs[band], 'f0': f0_med, 'f0_meow': f0m_med,
                'voiced_ratio': float(np.mean(voiced)), 'z': z}


def mfcc_dist(a, b):
    return float(np.linalg.norm(a['mfcc'] - b['mfcc']))


def ltas_dist(a, b):
    return float(np.sqrt(np.mean((a['ltas'] - b['ltas']) ** 2)))


def spk_cos(a, b):
    return float(a['z'] @ b['z'] / (np.linalg.norm(a['z']) * np.linalg.norm(b['z'])))


def semitones(a, b):
    return float(12 * np.log2(a / b))


def summarize(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
            'median': float(np.median(v)), 'n': int(v.size)}


def style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_distances(rows, out):
    """Per condition: d(src, ref) vs d(conv, ref), plus d(conv, src)."""
    metrics = [('mfcc', 'mean-MFCC distance (lower = closer)'), ('ltas', 'LTAS distance [dB] (lower = closer)'),
               ('spk', 'speaker-embedding cosine (higher = closer)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    rng = np.random.default_rng(0)
    for ax, (m, title) in zip(axes, metrics):
        kinds = [('src_ref', 'source ↔ target'), ('conv_ref', 'converted ↔ target'), ('conv_src', 'converted ↔ source')]
        for ci, cond in enumerate(CONDITIONS):
            sel = [r for r in rows if r['condition'] == cond]
            for ki, (k, _) in enumerate(kinds):
                xpos = ci * 4 + ki
                vals = np.array([r[f'{m}_{k}'] for r in sel])
                col = color_of(cond)
                alpha = [0.35, 1.0, 0.6][ki]
                ax.bar(xpos, vals.mean(), width=0.8, color=col, alpha=alpha,
                       hatch='//' if 'inter' in cond else None, edgecolor='white', linewidth=2)
                ax.scatter(xpos + rng.uniform(-0.2, 0.2, vals.size), vals, s=8, color=INK, alpha=0.5, zorder=3,
                           linewidths=0)
        ax.set_xticks([ci * 4 + 1 for ci in range(len(CONDITIONS))])
        ax.set_xticklabels([LABELS[c].replace(' ', '\n', 1) for c in CONDITIONS], fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        style(ax)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor='#8a8984', alpha=a, label=lab) for a, (_, lab) in
               zip([0.35, 1.0, 0.6], [('', 'source ↔ target (before)'), ('', 'converted ↔ target (after)'),
                                      ('', 'converted ↔ source')])]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Distances before / after conversion, by condition (bar = mean, dots = conversions; hatched = inter-gender)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / 'distances_by_condition.png', dpi=150)
    plt.close(fig)


def plot_gap_closed(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(1)
    for ax, (m, title) in zip(axes, [('mfcc', 'mean MFCC'), ('ltas', 'LTAS')]):
        for ci, cond in enumerate(CONDITIONS):
            vals = np.array([r[f'{m}_gap_closed'] for r in rows if r['condition'] == cond])
            ax.bar(ci, np.mean(vals), width=0.7, color=color_of(cond),
                   hatch='//' if 'inter' in cond else None, edgecolor='white', linewidth=2)
            ax.scatter(ci + rng.uniform(-0.18, 0.18, vals.size), vals, s=8, color=INK, alpha=0.5, zorder=3,
                       linewidths=0)
            ax.annotate(f'{np.mean(vals):.2f}', (ci, max(np.mean(vals), 0)), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=8.5, color=INK)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=':')
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([LABELS[c].replace(' ', '\n', 1) for c in CONDITIONS], fontsize=8.5)
        ax.set_title(f'Gap closed toward target — {title}', color=INK, fontsize=11, loc='left')
        style(ax)
    axes[0].set_ylabel('1 − d(conv,ref) / d(src,ref)', color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'gap_closed.png', dpi=150)
    plt.close(fig)


def plot_f0(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    markers = {'human2cat': 'o', 'intra': 'o', 'inter': '^'}
    for cond in CONDITIONS:
        sel = [r for r in rows if r['condition'] == cond and np.isfinite(r['f0_target']) and np.isfinite(r['f0_conv'])]
        mk = markers['human2cat'] if cond == 'human2cat' else markers[cond.split('_')[1]]
        ax.scatter([r['f0_target'] for r in sel], [r['f0_conv'] for r in sel], s=48, marker=mk,
                   color=color_of(cond), edgecolor='white', linewidth=1.5, label=LABELS[cond], zorder=3)
    lim = (60, 1600)
    ax.axhline(880, color=INK2, linewidth=0.8, linestyle='--')
    ax.annotate('model fo_range upper limit (880 Hz)', (62, 880), xytext=(0, 3), textcoords='offset points',
                fontsize=8, color=INK2)
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=':', label='output = target')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ticks = [62.5, 125, 250, 500, 1000]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_yticks(ticks); ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel('target reference median F0 [Hz]\n(cat targets: 150–2000 Hz search; humans: 60–1600 Hz)', color=INK2)
    ax.set_ylabel('converted output median F0 [Hz]', color=INK2)
    ax.set_title('Median F0: target vs converted (pYIN)', color=INK, fontsize=11, loc='left')
    style(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'f0_target_vs_output.png', dpi=150)
    plt.close(fig)


def plot_ltas(rows, feats, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    groups = [
        ('human sources', sorted({r['src'] for r in rows}), INK2, '-'),
        ('cat references', sorted({r['ref'] for r in rows if r['condition'] == 'human2cat'}), GROUP_COLOR['human2cat'], '--'),
        ('human→cat outputs', sorted({r['conv'] for r in rows if r['condition'] == 'human2cat'}), GROUP_COLOR['human2cat'], '-'),
        ('open-set outputs', sorted({r['conv'] for r in rows if r['condition'].startswith('open')}), GROUP_COLOR['open'], '-'),
    ]
    for name, files, col, ls in groups:
        L = np.mean([feats(f)['ltas'] for f in files], axis=0)
        fr = feats(files[0])['ltas_freqs']
        ax.plot(fr, L, color=col, linestyle=ls, linewidth=2, label=f'{name} (n={len(files)})')
    ax.set_xscale('log')
    ax.set_xlim(100, 8000)
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(['125', '250', '500', '1k', '2k', '4k', '8k'])
    ax.minorticks_off()
    ax.set_xlabel('frequency [Hz]', color=INK2)
    ax.set_ylabel('level re. band mean [dB]', color=INK2)
    ax.set_title('Long-term average spectrum (mean over files)', color=INK, fontsize=11, loc='left')
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'ltas.png', dpi=150)
    plt.close(fig)


def plot_spectrograms(rows, data_dir, out):
    """One source, its best and worst human->cat outputs (by MFCC distance), the cat targets, and an intra-gender output."""
    h2c = [r for r in rows if r['condition'] == 'human2cat']
    best = min(h2c, key=lambda r: r['mfcc_conv_ref'])
    src = best['src']
    intra = next(r for r in rows if r['src'] == src and r['condition'] == 'open_intra')
    panels = [(src, 'source (human)'), (best['ref'], f"target {best['ref_id']} (real cat)"),
              (best['conv'], f"converted → {best['ref_id']}"),
              (intra['ref'], f"target {intra['ref_id']} (human, intra)"),
              (intra['conv'], f"converted → {intra['ref_id']}")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.0 * len(panels)))
    for ax, (rel, title) in zip(axes, panels):
        x = librosa.resample(vc.load_audio(data_dir / rel), orig_sr=vc.SR, target_sr=FEAT_SR)[:FEAT_SR * 5]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=FEAT_SR, hop_length=160, x_axis='time', y_axis='hz', ax=ax,
                                 cmap='magma', vmin=-80, vmax=0)
        ax.set_title(f'{title}   [{rel}]', fontsize=9, loc='left', color=INK)
        ax.set_xlabel(''); ax.set_ylabel('Hz', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / 'spectrogram_examples.png', dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.data / 'manifest.csv') as f:
        manifest = list(csv.DictReader(f))

    model = vc.VoiceConverter()
    feats = Features(args.data, model)

    rows = []
    for i, m in enumerate(manifest):
        s, r, c = feats(m['src']), feats(m['ref']), feats(m['conv'])
        row = {k: m[k] for k in ('conv', 'src', 'ref', 'src_id', 'src_gender', 'ref_id', 'ref_kind',
                                 'ref_gender', 'condition')}
        for name, fn in (('mfcc', mfcc_dist), ('ltas', ltas_dist)):
            row[f'{name}_src_ref'], row[f'{name}_conv_ref'], row[f'{name}_conv_src'] = fn(s, r), fn(c, r), fn(c, s)
            row[f'{name}_gap_closed'] = 1 - row[f'{name}_conv_ref'] / row[f'{name}_src_ref']
        row['spk_src_ref'], row['spk_conv_ref'], row['spk_conv_src'] = spk_cos(s, r), spk_cos(c, r), spk_cos(c, s)
        row['f0_src'], row['f0_ref'], row['f0_conv'] = s['f0'], r['f0'], c['f0']
        row['f0_ref_meow'] = r['f0_meow']
        row['f0_target'] = r['f0_meow'] if m['ref_kind'] == 'cat' else r['f0']
        row['f0_err_st'] = abs(semitones(c['f0'], row['f0_target']))
        row['f0_shift_st'] = semitones(c['f0'], s['f0'])
        row['f0_needed_st'] = semitones(row['f0_target'], s['f0'])
        fo_model = np.load(args.data / 'conv_fo' / (pathlib.Path(m['conv']).stem + '.npy'))
        row['f0_model_pred'] = float(np.median(fo_model[fo_model > 0])) if (fo_model > 0).any() else float('nan')
        row['voiced_ratio_conv'], row['voiced_ratio_ref'] = c['voiced_ratio'], r['voiced_ratio']
        rows.append(row)
        print(f'[{i + 1}/{len(manifest)}] {m["conv"]}', flush=True)

    with open(args.out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    # Spread among the real references themselves (scale for the distances above)
    refs = sorted({(m['ref'], m['ref_kind']) for m in manifest})
    spread_rows = []
    for (a, ka), (b, kb) in itertools.combinations(refs, 2):
        pair = 'cat-cat' if ka == kb == 'cat' else ('cat-human' if 'cat' in (ka, kb) else 'human-human')
        fa, fb = feats(a), feats(b)
        spread_rows.append({'a': a, 'b': b, 'pair': pair, 'mfcc': mfcc_dist(fa, fb), 'ltas': ltas_dist(fa, fb),
                            'spk_cos': spk_cos(fa, fb)})
    with open(args.out / 'reference_spread.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(spread_rows[0]))
        w.writeheader()
        w.writerows(spread_rows)

    metric_keys = [k for k in rows[0] if k not in ('conv', 'src', 'ref', 'src_id', 'src_gender', 'ref_id',
                                                   'ref_kind', 'ref_gender', 'condition')]
    summary = {'conditions': {}, 'reference_spread': {}}
    summary_rows = []
    for cond in CONDITIONS:
        sel = [r for r in rows if r['condition'] == cond]
        summary['conditions'][cond] = {k: summarize([r[k] for r in sel]) for k in metric_keys}
        summary_rows.append({'condition': cond, 'n': len(sel),
                             **{k: f"{summary['conditions'][cond][k]['mean']:.3f} ± {summary['conditions'][cond][k]['std']:.3f}"
                                for k in metric_keys}})
    for pair in ('cat-cat', 'human-human', 'cat-human'):
        sel = [r for r in spread_rows if r['pair'] == pair]
        summary['reference_spread'][pair] = {k: summarize([r[k] for r in sel]) for k in ('mfcc', 'ltas', 'spk_cos')}
    summary['features'] = {'sr_hz': FEAT_SR, 'model_step': int(manifest[0]['model_step'])}
    with open(args.out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(args.out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    import librosa.display  # noqa: F401  (registers specshow)
    plot_distances(rows, args.out)
    plot_gap_closed(rows, args.out)
    plot_f0(rows, args.out)
    plot_ltas(rows, feats, args.out)
    plot_spectrograms(rows, args.data, args.out)

    # console digest
    keys = ['mfcc_src_ref', 'mfcc_conv_ref', 'mfcc_gap_closed', 'ltas_gap_closed', 'spk_src_ref', 'spk_conv_ref',
            'f0_target', 'f0_conv', 'f0_model_pred', 'f0_err_st']
    print('condition'.ljust(14) + ''.join(k.rjust(17) for k in keys))
    for cond in CONDITIONS:
        print(cond.ljust(14) + ''.join(f"{summary['conditions'][cond][k]['mean']:17.3f}" for k in keys))
    for pair, d in summary['reference_spread'].items():
        print(f"spread {pair}: mfcc {d['mfcc']['mean']:.2f}, ltas {d['ltas']['mean']:.2f}, spk_cos {d['spk_cos']['mean']:.3f}")


if __name__ == '__main__':
    main()
