"""Subexperiment 03 — analysis: where is cat F0 lost, F0 generation or decoder?

Reads data/ from generate.py. All pitch errors are in semitones (st) and
evaluated on "truth frames": frames where the input is reliably voiced
(pYIN voiced, prob > 0.5; for cats also F0 >= 200 Hz).

Stages (per clip, then aggregated per group):
  generation   cond F0 vs true F0                       (setting zq; z = without quantizer)
  decoder      output F0 vs cond F0                     (does synth render what it is given?)
  oracle       output F0 vs injected true F0            (oracle_fo, oracle_fo_vf)
  end-to-end   output F0 vs true F0                     (setting zq)
Accuracy = fraction of frames with |error| < 1 st; frames where the output is
unvoiced (pYIN voiced decision) count as errors (the pitch was not rendered).
Octave errors = frames with |error -/+ 12| < 1 st.
The decoder has a latency of a few frames; it is estimated per clip (0-5
frames, maximizing output-vs-cond agreement in setting zq) and every output
of that clip is shifted back by it before comparison.

Generation is further decomposed, using fo = 10 * 2^((mu + sigma * s) / 12):
  mean offset  median(mu) - median(true st)   speaker-code term
  contour r    Pearson r of predicted vs true st over truth frames
  range ratio  std(predicted st) / std(true st)
  clipped      fraction of truth frames where cond F0 sits at the 880 Hz cap

Supplemental decoder pitch sweep (data/sweep from generate_sweep.py): true
F0 injected shifted by k semitones; rendering accuracy is binned by the
absolute injected F0, separately for cat and human conditioning.

Outputs: results/clip_metrics.csv, results/summary.csv, results/summary.json,
         results/oracle_by_f0_band.csv, results/sweep_by_f0_band.csv,
         results/*.png
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
SR, HOP = 24000, 240
PYIN = dict(fmin=50.0, fmax=2000.0, sr=SR, frame_length=2048, hop_length=HOP)
FO_CAP = 880.0
GROUPS = ['cat', 'human_closed', 'human_open']
GROUP_LABEL = {'cat': 'cat', 'human_closed': 'human closed-set (jvs)', 'human_open': 'human open-set (CV)'}
# Reference palette (dataviz skill), first three slots; same color roles as subexp 01
GROUP_COLOR = {'cat': '#eb6834', 'human_closed': '#1baf7a', 'human_open': '#2a78d6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SETTINGS = ('zq', 'z', 'oracle_fo', 'oracle_fo_vf')
F0_BANDS = [(200, 440), (440, 880), (880, 2000)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=pathlib.Path, default=HERE / 'data')
    p.add_argument('--out', type=pathlib.Path, default=HERE / 'results')
    return p.parse_args()


def st(f):
    """Hz -> semitones re 10 Hz (the model's internal scale)."""
    return 12 * np.log2(np.asarray(f) / 10.0)


MAX_LAG = 5


def pitch(x):
    """Output F0: pYIN voiced decision only (the stricter prob > 0.5 is used for ground truth)."""
    f0, voiced, _ = librosa.pyin(x, **PYIN)
    return np.where(voiced, f0, np.nan)


def shift(f, lag):
    """Undo a decoder latency of `lag` frames."""
    return np.concatenate((f[lag:], np.full(lag, np.nan))) if lag else f


def decoder_lag(f_out, fo_c, voiced_c):
    scores = []
    for lag in range(MAX_LAG + 1):
        o = shift(f_out, lag)
        ok = voiced_c & np.isfinite(o)
        scores.append(np.mean(np.abs(st(o[ok]) - st(fo_c[ok])) < 1) if ok.sum() > 20 else -1)
    return int(np.argmax(scores))


def log_flatness(x, frames):
    """Median log10 spectral flatness (0 = white noise, more negative = more tonal) over `frames`."""
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=HOP)) ** 2
    fl = librosa.feature.spectral_flatness(S=S)[0]
    n = min(fl.size, frames.size)
    sel = frames[:n]
    return float(np.median(np.log10(fl[:n][sel] + 1e-12))) if sel.any() else float('nan')


def octave_rate(err):
    e = err[np.isfinite(err)]
    return float(np.mean(np.abs(np.abs(e) - 12) < 1)) if err.size else float('nan')


def acc(err):
    """err: st errors with NaN where the output was unvoiced."""
    return float(np.mean(np.abs(np.nan_to_num(err, nan=99.0)) < 1.0)) if err.size else float('nan')


def med(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float('nan')


def safe_corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float('nan')
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def style(ax, grid_x=False):
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    if grid_x:
        ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def log_hz_axis(ax, which, lim=(50, 2000)):
    ticks = [62.5, 125, 250, 500, 1000, 2000]
    getattr(ax, f'set_{which}scale')('log')
    getattr(ax, f'set_{which}lim')(lim)
    getattr(ax, f'set_{which}ticks')(ticks)
    getattr(ax, f'set_{which}ticklabels')([f'{t:g}' for t in ticks])
    ax.minorticks_off()


def analyze_clip(row, data):
    fr = np.load(data / 'frames' / f"{row['clip']}.npz")
    T = fr['mask_true']
    f_true = fr['f0_true']
    st_true = st(f_true)
    x_in = librosa.load(str(data / 'input' / f"{row['clip']}.wav"), sr=SR)[0]
    out = {s: librosa.load(str(data / 'output' / f"{row['clip']}__{s}.wav"), sr=SR)[0] for s in SETTINGS}
    f_out = {}
    for s, y in out.items():
        f = pitch(y)
        f_out[s] = np.pad(f, (0, max(0, T.size - f.size)), constant_values=np.nan)[:T.size]
    lag = decoder_lag(f_out['zq'], fr['zq_fo'], fr['zq_vf'] > 0.5)
    f_out = {s: shift(f, lag) for s, f in f_out.items()}

    r = {k: row[k] for k in ('clip', 'group', 'gender', 'code')}
    r['z_to_zq_dist'] = float(row['z_to_zq_dist'])
    r['decoder_lag_frames'] = lag
    r['n_truth'] = int(T.sum())
    r['f0_true_med'] = med(f_true[T])
    r['f0_true_p95'] = float(np.nanpercentile(f_true[T], 95))
    r['frac_true_above_cap'] = float(np.mean(f_true[T] > FO_CAP))

    for s in ('zq', 'z'):
        fo_c = fr[f'{s}_fo']
        mu = fr[f'{s}_fo_semitone_mu']
        vf = fr[f'{s}_vf']
        st_c = st(fo_c)
        e_gen = st_c[T] - st_true[T]
        r[f'{s}_gen_acc'] = acc(e_gen)
        r[f'{s}_gen_bias_st'] = med(e_gen)
        r[f'{s}_gen_abs_st'] = med(np.abs(e_gen))
        r[f'{s}_mu_hz'] = float(10 * 2 ** (np.median(mu[T]) / 12))
        r[f'{s}_mean_offset_st'] = float(np.median(mu[T]) - np.median(st_true[T]))
        sigma = np.minimum(np.exp(0.5 * np.minimum(fr[f'{s}_fo_semitone_sigma_logvar'], 12.0)), np.e ** 6)
        r[f'{s}_sigma_st'] = float(np.median(sigma[T]))
        r[f'{s}_contour_r'] = safe_corr(st_c[T], st_true[T])
        r[f'{s}_range_ratio'] = float(np.std(st_c[T]) / np.std(st_true[T])) if np.std(st_true[T]) > 0 else float('nan')
        r[f'{s}_clipped'] = float(np.mean(fo_c[T] >= FO_CAP - 1))
        r[f'{s}_cond_f0_med'] = med(fo_c[T])
        r[f'{s}_vf_on_truth'] = float(np.mean(vf[T] > 0.5))
        # decoder: output vs what it was given, on frames the condition marks voiced
        V = vf > 0.5
        e_dec = st(f_out[s][V]) - st_c[V]
        r[f'{s}_dec_acc'] = acc(e_dec)
        r[f'{s}_dec_bias_st'] = med(e_dec)
        r[f'{s}_dec_octave'] = octave_rate(e_dec)
        r[f'{s}_out_voiced_on_vf'] = float(np.mean(np.isfinite(f_out[s][V]))) if V.any() else float('nan')
        # end to end
        e_tot = st(f_out[s][T]) - st_true[T]
        r[f'{s}_e2e_acc'] = acc(e_tot)
        r[f'{s}_e2e_bias_st'] = med(e_tot)
        r[f'{s}_out_f0_med'] = med(f_out[s][T])
        r[f'{s}_beta_med'] = float(np.median(fr[f'{s}_beta'][T]))
    for s in ('oracle_fo', 'oracle_fo_vf'):
        e = st(f_out[s][T]) - st_true[T]
        r[f'{s}_acc'] = acc(e)
        r[f'{s}_bias_st'] = med(e)
        r[f'{s}_octave'] = octave_rate(e)
        r[f'{s}_out_voiced'] = float(np.mean(np.isfinite(f_out[s][T])))
        r[f'{s}_out_f0_med'] = med(f_out[s][T])
    r['logflat_in'] = log_flatness(x_in, T)
    r['logflat_zq'] = log_flatness(out['zq'], T)
    r['logflat_oracle_fo_vf'] = log_flatness(out['oracle_fo_vf'], T)

    frames = {'f_true': f_true[T], 'fo_zq': fr['zq_fo'][T], 'mu_zq': fr['zq_fo_semitone_mu'][T],
              **{f'out_{s}': f_out[s][T] for s in SETTINGS}}
    series = {'f_true': f_true, 'mask': T, 'fo_zq': fr['zq_fo'], 'vf_zq': fr['zq_vf'],
              'mu_zq_hz': 10 * 2 ** (fr['zq_fo_semitone_mu'] / 12), **{f'out_{s}': f_out[s] for s in SETTINGS}}
    return r, frames, series


SWEEP_EDGES = np.array([100, 141, 200, 283, 400, 566, 800, 1131, 1600, 2263])


def analyze_sweep(rows, data):
    """Frame-level (injected F0, error) pairs from the pitch sweep, pooled per group."""
    pooled = {'cat': ([], []), 'human': ([], [])}
    for r in rows:
        kind = 'cat' if r['group'] == 'cat' else 'human'
        fr = np.load(data / 'frames' / f"{r['clip']}.npz")
        T, f_true = fr['mask_true'], fr['f0_true']
        for path in sorted((data / 'sweep').glob(f"{r['clip']}__*.wav")):
            k = int(path.stem.split('__')[1])
            f = pitch(librosa.load(str(path), sr=SR)[0])
            f = shift(np.pad(f, (0, max(0, T.size - f.size)), constant_values=np.nan)[:T.size],
                      r['decoder_lag_frames'])
            inj = f_true[T] * 2 ** (k / 12)
            pooled[kind][0].append(inj)
            pooled[kind][1].append(st(f[T]) - st(inj))
    out = []
    for kind, (inj, err) in pooled.items():
        inj, err = np.concatenate(inj), np.concatenate(err)
        for lo, hi in zip(SWEEP_EDGES[:-1], SWEEP_EDGES[1:]):
            sel = (inj >= lo) & (inj < hi)
            if sel.sum() < 50:
                continue
            e = err[sel]
            out.append({'group': kind, 'band_lo_hz': int(lo), 'band_hi_hz': int(hi),
                        'center_hz': round(float(np.sqrt(lo * hi)), 1), 'n_frames': int(sel.sum()),
                        'acc': round(acc(e), 4), 'voiced': round(float(np.mean(np.isfinite(e))), 4),
                        'acc_if_voiced': round(acc(e[np.isfinite(e)]), 4) if np.isfinite(e).any() else float('nan'),
                        'octave': round(octave_rate(e), 4)})
    return out


def plot_sweep(band_rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    colors = {'cat': GROUP_COLOR['cat'], 'human': GROUP_COLOR['human_closed']}
    labels = {'cat': 'cat conditioning (F0 shifted 0 to −24 st)', 'human': 'human conditioning (F0 shifted 0 to +24 st)'}
    for ax, (key, title) in zip(axes, [('acc', 'rendered within ±1 st'), ('voiced', 'output voiced'),
                                       ('octave', 'octave errors (among voiced)')]):
        for g in ('cat', 'human'):
            sel = [b for b in band_rows if b['group'] == g]
            ax.plot([b['center_hz'] for b in sel], [b[key] for b in sel], color=colors[g], linewidth=2,
                    marker='o', markersize=6, markeredgecolor='white', label=labels[g])
        ax.axvline(FO_CAP, color=INK2, linestyle='--', linewidth=0.8)
        log_hz_axis(ax, 'x', lim=(100, 2263))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, color=INK, fontsize=10.5, loc='left')
        ax.set_xlabel('injected F0 [Hz]', color=INK2)
        style(ax, grid_x=True)
    axes[0].set_ylabel('fraction of frames', color=INK2)
    axes[0].annotate('880 Hz', (FO_CAP, 0.97), xytext=(3, 0), textcoords='offset points', fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Decoder pitch sweep: true F0 and voicing injected, F0 shifted by k semitones', fontsize=11,
                 color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'decoder_pitch_sweep.png', dpi=150)
    plt.close(fig)


STAGES = [
    ('zq_gen_acc', 'generation\n(cond F0 vs true)'),
    ('z_gen_acc', 'generation,\nno quantizer (z)'),
    ('zq_dec_acc', 'decoder\n(output vs cond F0)'),
    ('oracle_fo_acc', 'decoder with\noracle F0'),
    ('oracle_fo_vf_acc', 'decoder with\noracle F0 + voicing'),
    ('zq_e2e_acc', 'end-to-end\n(output vs true)'),
]


def plot_stage_accuracy(rows, out):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.26
    rng = np.random.default_rng(0)
    for gi, g in enumerate(GROUPS):
        sel = [r for r in rows if r['group'] == g]
        for si, (k, _) in enumerate(STAGES):
            vals = np.array([r[k] for r in sel])
            xpos = si + (gi - 1) * w
            ax.bar(xpos, np.mean(vals), width=w - 0.03, color=GROUP_COLOR[g], label=GROUP_LABEL[g] if si == 0 else None)
            ax.scatter(xpos + rng.uniform(-w / 5, w / 5, vals.size), vals, s=7, color=INK, alpha=0.45, zorder=3,
                       linewidths=0)
            if g == 'cat':
                ax.annotate(f'{np.mean(vals):.2f}', (xpos, np.mean(vals)), textcoords='offset points', xytext=(0, 3),
                            ha='center', fontsize=8, color=INK)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([lab for _, lab in STAGES], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('fraction of truth frames within ±1 semitone', color=INK2)
    ax.set_title('F0 accuracy by stage (bar = mean over clips, dots = clips)', color=INK, fontsize=11, loc='left')
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=3)
    fig.tight_layout()
    fig.savefig(out / 'stage_accuracy.png', dpi=150)
    plt.close(fig)


def plot_f0_levels(rows, out):
    panels = [('zq_mu_hz', 'speaker mean μ(z_q)'), ('zq_cond_f0_med', 'generated F0 (condition)'),
              ('zq_out_f0_med', 'output F0 (normal)'), ('oracle_fo_vf_out_f0_med', 'output F0 (oracle F0 + voicing)')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    for ax, (k, title) in zip(axes, panels):
        for g in GROUPS:
            sel = [r for r in rows if r['group'] == g and np.isfinite(r[k])]
            ax.scatter([r['f0_true_med'] for r in sel], [r[k] for r in sel], s=36, color=GROUP_COLOR[g],
                       edgecolor='white', linewidth=1.2, label=GROUP_LABEL[g], zorder=3)
        ax.plot((50, 2000), (50, 2000), color=INK2, linestyle=':', linewidth=1)
        ax.axhline(FO_CAP, color=INK2, linestyle='--', linewidth=0.8)
        log_hz_axis(ax, 'x'); log_hz_axis(ax, 'y')
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('true median F0 of the input [Hz]', color=INK2, fontsize=9)
        style(ax, grid_x=True)
    axes[0].set_ylabel('median F0 [Hz]', color=INK2)
    axes[0].annotate('880 Hz cap', (55, FO_CAP), xytext=(0, 3), textcoords='offset points', fontsize=8, color=INK2)
    axes[-1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Per-clip median F0 along the pipeline (dotted = identity)', fontsize=11, color=INK, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'f0_along_pipeline.png', dpi=150)
    plt.close(fig)


def plot_oracle_hist(frames, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    bins = np.geomspace(50, 2000, 60)
    specs = [('cat', 'fo_zq', 'cat: generated F0 vs true'), ('cat', 'out_oracle_fo_vf', 'cat: output vs injected true F0'),
             ('human', 'out_oracle_fo_vf', 'human: output vs injected true F0')]
    for ax, (grp, key, title) in zip(axes, specs):
        fs = [f for f in frames if (f['group'] == 'cat') == (grp == 'cat')]
        xt = np.concatenate([f['f_true'] for f in fs])
        yv = np.concatenate([f[key] for f in fs])
        ok = np.isfinite(yv)
        H, _, _ = np.histogram2d(xt[ok], yv[ok], bins=[bins, bins])
        ax.pcolormesh(bins, bins, np.log1p(H.T), cmap='Greys', shading='flat')
        ax.plot((50, 2000), (50, 2000), color='#eb6834', linestyle=':', linewidth=1.2)
        ax.axhline(FO_CAP, color=INK2, linestyle='--', linewidth=0.8)
        log_hz_axis(ax, 'x'); log_hz_axis(ax, 'y')
        ax.set_title(f'{title}\n({(~ok).mean():.0%} of frames unvoiced in output, not shown)', fontsize=9.5,
                     color=INK, loc='left')
        ax.set_xlabel('true F0 [Hz]', color=INK2, fontsize=9)
        style(ax, grid_x=True)
    axes[0].set_ylabel('F0 [Hz]', color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'frame_f0_histograms.png', dpi=150)
    plt.close(fig)


def plot_contours(series, rows, out):
    cats = sorted([r for r in rows if r['group'] == 'cat'], key=lambda r: -r['f0_true_med'])
    picks = [cats[0], cats[len(cats) // 2]]
    human = next(r for r in rows if r['group'] == 'human_closed')
    picks.append(human)
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.8 * len(picks)))
    for ax, r in zip(axes, picks):
        s = series[r['clip']]
        t = np.arange(s['f_true'].size) * HOP / SR
        ax.plot(t, np.where(s['mask'], s['f_true'], np.nan), color=INK, linewidth=2.2, label='true F0 (input)')
        ax.plot(t, np.where(s['vf_zq'] > 0.5, s['fo_zq'], np.nan), color='#2a78d6', linewidth=1.5,
                label='generated F0 (cond, voiced)')
        ax.plot(t, s['mu_zq_hz'], color='#2a78d6', linewidth=1, linestyle='--', label='speaker mean μ')
        ax.plot(t, s['out_zq'], color='#eb6834', linewidth=1.2, label='output F0 (normal)')
        ax.plot(t, s['out_oracle_fo_vf'], color='#1baf7a', linewidth=1.2, label='output F0 (oracle F0+voicing)')
        ax.axhline(FO_CAP, color=INK2, linestyle='--', linewidth=0.8)
        log_hz_axis(ax, 'y')
        ax.set_xlim(0, t[-1])
        ax.set_title(f"{r['clip']} ({GROUP_LABEL[r['group']]}, true median {r['f0_true_med']:.0f} Hz)",
                     fontsize=9.5, color=INK, loc='left')
        ax.set_ylabel('Hz', color=INK2)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc='lower left')
    axes[-1].set_xlabel('time [s]', color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'contour_examples.png', dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.data / 'manifest.csv') as f:
        manifest = list(csv.DictReader(f))

    rows, frames, series = [], [], {}
    for i, m in enumerate(manifest):
        r, fr, se = analyze_clip(m, args.data)
        rows.append(r)
        frames.append({'group': m['group'], **fr})
        series[m['clip']] = se
        print(f"[{i + 1}/{len(manifest)}] {m['clip']}", flush=True)

    with open(args.out / 'clip_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    keys = [k for k in rows[0] if k not in ('clip', 'group', 'gender', 'code')]
    summary = {'groups': {}, 'codes': {}}
    srows = []
    for g in GROUPS:
        sel = [r for r in rows if r['group'] == g]
        summary['groups'][g] = {k: {'mean': float(np.nanmean([r[k] for r in sel])),
                                    'median': float(np.nanmedian([r[k] for r in sel])),
                                    'n': len(sel)} for k in keys}
        srows.append({'group': g, 'n_clips': len(sel), **{k: round(summary['groups'][g][k]['mean'], 4) for k in keys}})
        summary['codes'][g] = sorted({int(r['code']) for r in sel})
    cat_codes, human_codes = set(summary['codes']['cat']), set(summary['codes']['human_closed']) | set(summary['codes']['human_open'])
    summary['codes']['cat_codes_also_used_by_humans'] = sorted(cat_codes & human_codes)

    # oracle decoder accuracy by true-F0 band (pooled frames)
    band_rows = []
    for grp_name, is_cat in (('cat', True), ('human', False)):
        fs = [f for f in frames if (f['group'] == 'cat') == is_cat]
        ft = np.concatenate([f['f_true'] for f in fs])
        for lo, hi in F0_BANDS + [(50, 200)]:
            sel = (ft >= lo) & (ft < hi)
            if sel.sum() < 50:
                continue
            row = {'group': grp_name, 'band_hz': f'{lo}-{hi}', 'n_frames': int(sel.sum())}
            for key in ('fo_zq', 'out_zq', 'out_oracle_fo', 'out_oracle_fo_vf'):
                v = np.concatenate([f[key] for f in fs])[sel]
                e = st(v) - st(ft[sel])
                row[f'{key}_acc'] = round(acc(e), 4)
                row[f'{key}_voiced'] = round(float(np.mean(np.isfinite(v))), 4)
                # accuracy among frames the output does voice: wrong pitch vs not rendered
                row[f'{key}_acc_if_voiced'] = round(acc(e[np.isfinite(e)]), 4)
                row[f'{key}_octave'] = round(octave_rate(e), 4)
            band_rows.append(row)
    with open(args.out / 'oracle_by_f0_band.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(band_rows[0]))
        w.writeheader()
        w.writerows(band_rows)
    summary['oracle_by_f0_band'] = band_rows

    if (args.data / 'sweep').is_dir():
        sweep_rows = analyze_sweep(rows, args.data)
        with open(args.out / 'sweep_by_f0_band.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
            w.writeheader()
            w.writerows(sweep_rows)
        summary['sweep_by_f0_band'] = sweep_rows
        plot_sweep(sweep_rows, args.out)
        for b in sweep_rows:
            print('sweep', b)

    with open(args.out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(args.out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    plot_stage_accuracy(rows, args.out)
    plot_f0_levels(rows, args.out)
    plot_oracle_hist(frames, args.out)
    plot_contours(series, rows, args.out)

    show = ['f0_true_med', 'zq_mu_hz', 'zq_cond_f0_med', 'zq_out_f0_med', 'zq_mean_offset_st', 'zq_contour_r',
            'zq_range_ratio', 'zq_clipped', 'zq_vf_on_truth', 'zq_gen_acc', 'z_gen_acc', 'zq_dec_acc',
            'zq_dec_octave', 'zq_out_voiced_on_vf', 'oracle_fo_acc', 'oracle_fo_vf_acc', 'oracle_fo_vf_octave',
            'oracle_fo_vf_out_voiced', 'zq_e2e_acc', 'decoder_lag_frames', 'z_to_zq_dist', 'frac_true_above_cap',
            'zq_beta_med', 'logflat_in', 'logflat_zq', 'logflat_oracle_fo_vf']
    print('metric'.ljust(26) + ''.join(g.rjust(15) for g in GROUPS))
    for k in show:
        print(k.ljust(26) + ''.join(f"{summary['groups'][g][k]['mean']:15.3f}" for g in GROUPS))
    for b in band_rows:
        print(b)
    print('codes', summary['codes'])


if __name__ == '__main__':
    main()
