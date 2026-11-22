# 03. Acoustic similarity analysis: where is cat F0 lost?

## Objective

With the v1 model, subexperiment 01 found that human→cat outputs kept a human pitch: about 228 Hz against cat
targets of about 546 Hz, an error of about 16 semitones. This subexperiment finds **which stage of the model loses
the cat F0**, and checks whether the pitch-controllable v2 model fixes it. It also compares voicing, aperiodicity
(decay rate β, spectral flatness) and the speaker-code quantization of real cats against real humans.

### Model versions and their F0 stages

| | v1 (previous run) | **v2 (this report)** |
|---|---|---|
| checkpoint | `~/Documents/Repos/nvcnsf_app/app_py/checkpoints/`, step 500000 | `/path/to/checkpoint/`, step 100000 |
| F0 stage | **F0 generation** (`Condition`): `fo = 10 · 2^((μ(z) + σ(z) · s(c)) / 12)`, where the contour `s` comes from the content features | **F0 extractor**: reads F0 from the input waveform; the speaker code only gives semitone statistics μ(z), σ(z) for mapping to a target speaker (`map_fo`), plus an optional `semitone_shift` |
| F0 range (`fo_range`) | 27.5–880 Hz | 27.5–1760 Hz |
| decoder | cyclic-noise source driven by `fo` + neural filters | same structure (`decode`), retrained |

In self-reconstruction (source code = target code) the v2 mapping is the identity, so the F0 stage does not depend
on `z`. The output column names are kept from v1: `gen` is the F0 stage (now the extractor) and `mu` is μ(z). The v1
outputs are kept in `results_v1_old/`, and every v1 number below comes from there.

## Method

### Self-reconstruction design

Every clip is reconstructed with **its own** speaker code. Input and output are then time-aligned, so the input's
pYIN F0 serves as frame-level ground truth for every stage. Four settings per clip (`generate.py`):

| setting | speaker code | F0 / voicing given to the decoder | isolates |
|---|---|---|---|
| `zq` | `z_q` (normal) | model | normal operation |
| `z` | continuous `z` | model | effect of the speaker-code quantizer (v2: no effect on F0 by construction) |
| `oracle_fo` | `z_q` | **true F0**; model voicing | decoder alone |
| `oracle_fo_vf` | `z_q` | **true F0 and true voicing** | decoder alone |

**Supplemental pitch sweep** (`generate_sweep.py`). The oracle runs inject cat F0 under cat conditioning, so a
decoder failure could come from the frequency or from the cat code. To separate these, each clip is re-synthesized
with its true F0 and voicing, but F0 shifted by k semitones: humans by k = 0…+24 (into the cat range) and cats by
k = 0…−24 (into the human range). If accuracy depends only on the injected frequency, both groups fall on one
curve.

### Data

Unchanged from v1 (same seed, same clips):

| group | n | source |
|---|---|---|
| cat | 16 | meow-10k, screened from 300 random clips as in subexp 01 (top pYIN voiced ratio, median F0 250–1500 Hz); silence-trimmed, ≤ 8 s |
| human closed-set | 8 (4 M, 4 F) | jvs005, 006, 009, 011 (M) and jvs007, 008, 010, 014 (F), `VOICEACTRESS100_001` (training speakers; disjoint from the subexp-01 references) |
| human open-set | 8 (4 M, 4 F) | Common Voice ja test, the same 8 sources as subexp 01 |

### Processing and measures

- `model.py` runs the v2 stages block-wise **with the streaming state carried across blocks**, as in `infer.py`. A
  single whole-utterance call needs more than 11 GB of GPU memory. The stages are content encoder, F0 extractor,
  semitone statistics and `map_fo`, source parameters (vf, α, β), mgc, and decode. The decoder inputs can be
  replaced for the oracle settings.
- Ground truth comes from pYIN (50–2000 Hz, 10 ms hop). "Truth frames" are frames where pYIN says voiced with
  probability > 0.5; for cats the F0 must also be ≥ 200 Hz, which removes background noise (see subexp 01).
- Output F0 uses pYIN's voiced decision. Outputs are shifted back by the decoder latency, estimated per clip
  (mean 1.9–2.1 frames).
- **Accuracy** is the fraction of truth frames whose pitch is within ±1 semitone. An unvoiced output frame counts as
  wrong (the pitch was not rendered). Octave errors are reported separately.

## Results

### 1. Accuracy per stage

These are means over clips (`results/summary.csv`, `stage_accuracy.png`).

| stage | cat v2 | human closed v2 | human open v2 | cat v1 | human closed v1 | human open v1 |
|---|---|---|---|---|---|---|
| F0 stage vs true F0 (v1: generation, v2: extractor) | **0.85** | 0.93 | 0.94 | 0.13 | 0.28 | 0.21 |
| decoder: output vs given F0 | 0.95 | 0.93 | 0.94 | 0.72 | 0.70 | 0.54 |
| decoder with oracle F0 | 0.99 | 0.99 | 0.99 | — | — | — |
| decoder with oracle F0 + voicing | **0.995** | 0.97 | 0.93 | 0.38 | 0.59 | 0.46 |
| end to end: output vs true F0 | **0.84** | 0.95 | 0.95 | 0.12 | 0.19 | 0.17 |

**Both stages that failed in v1 now work.** End-to-end cat pitch accuracy rose from 0.12 to 0.84. Human accuracy
also rose (0.17–0.19 → 0.95), so the old pipeline was weak for everyone and especially for cats. Cats still trail
humans by about 0.1, and that loss is in the F0 extractor. The decoder renders what it is given (0.95) and renders
the true cat F0 almost perfectly (0.995).

### 2. The F0 stage (`f0_along_pipeline.png`, `contour_examples.png`)

| component | cat v2 | human closed v2 | human open v2 | cat v1 |
|---|---|---|---|---|
| true median F0 [Hz] | 572 | 228 | 168 | 572 |
| F0-stage median F0 [Hz] | 576 | 229 | 169 | 310 |
| contour r (F0 stage vs true, st) | **0.76** | 0.98 | 0.92 | 0.04 |
| range ratio (std F0 stage / std true) | 1.45 | 1.06 | 1.19 | 0.89 |
| frames at the F0 cap | 0 (cap 1760 Hz) | 0 | 0 | 0 (cap 880 Hz) |
| voicing (vf > 0.5) on truth frames | 0.97 | 0.97 | 0.95 | 0.99 |

- **The cat contour is now carried** (r = 0.76, v1 0.04). In v1 the contour came from content features that did not
  encode a cat's pitch glide. The v2 extractor reads it from the waveform.
- **The extractor's errors are concentrated at high pitch.** By true-F0 band (pooled cat frames,
  `oracle_by_f0_band.csv`):

  | true F0 | frames | extractor ±1 st | end to end ±1 st | decoder with oracle F0 + voicing | v1 decoder with oracle F0 + voicing |
  |---|---|---|---|---|---|
  | 200–440 Hz | 1096 | 0.92 | 0.92 | 0.99 | 0.93 |
  | 440–880 Hz | 4284 | 0.88 | 0.88 | 1.00 | 0.22 |
  | 880–2000 Hz | 569 | **0.33** | 0.30 | 0.99 | 0.00 |

  About 10 % of cat voiced frames are above 880 Hz. Per clip, the lowest accuracies are cat05 (median 1237 Hz,
  0.29) and cat04 (819 Hz, 0.61). Two lower-pitched noisy clips are also low: cat07 (543 Hz, 0.57) and cat02
  (610 Hz, 0.72). The other 12 cats reach 0.81–0.99. The extractor's range ratio of 1.45 for cats means its contour
  swings more widely than the true one; this is consistent with jumps at high pitch.
- **The speaker statistics μ(z) are still too low for cats.**

  | | cat v2 | human closed v2 | human open v2 | cat v1 |
  |---|---|---|---|---|
  | μ(z) [Hz] | 367 | 208 | 193 | 243 |
  | offset from true median [st] | **−7.7** | −2.3 | +3.0 | −14.5 |

  The bias halved from v1, but it is still about 3× the human offsets. It does not affect self-reconstruction (the
  mapping is the identity). It does affect conversion *to* a cat: `map_fo` would put human speech about 8 st below
  the cat's real pitch. This is why subexp 01 v2 needed a per-pair `semitone_shift` (+9 st on average).
- **Quantization.** The continuous `z` gives the same μ, so the quantizer is not the cause. The 16 cats use only 4
  codes (119, 276, 359, 455), and two of these (359, 455) are also used by human speakers. The distance from `z` to
  the nearest code is larger for cats (7.3 vs 5.8 for humans), so cat embeddings are still out of distribution.

### 3. The decoder (`decoder_pitch_sweep.png`, `sweep_by_f0_band.csv`, `frame_f0_histograms.png`)

Pitch sweep, accuracy (±1 st) by injected F0 band; v1 values are accuracy among voiced output frames:

| injected F0 | cat conditioning v2 | human conditioning v2 | cat v1 | human v1 |
|---|---|---|---|---|
| 100–141 Hz | 0.99 | 0.93 | — | octave errors (64–70 %) |
| 283–400 Hz | 0.99 | 0.98 | 0.92 | 0.90 |
| 400–566 Hz | 1.00 | 0.98 | 0.82 | 0.42 |
| 566–800 Hz | 1.00 | 0.97 | **0.01** | **0.00** |
| 800–1131 Hz | 0.99 | 0.97 | never voiced | 0.00 |
| 1131–1600 Hz | 1.00 | 0.97 | never voiced | 0.00 |
| 1600–2263 Hz | 1.00 (50 frames) | 1.00 (150 frames) | — | — |

- **The v1 decoder pitch ceiling (~550 Hz) is gone.** v2 renders injected F0 from 100 Hz to about 2 kHz with
  ≥ 0.93 accuracy and < 1 % octave errors, under both cat and human conditioning. v1 collapsed above 566 Hz and was
  silent above 880 Hz under cat conditioning.
- **The low-pitch octave errors of v1 are also gone.** v1 rendered 64–70 % of voiced frames below 140 Hz an
  octave too low; v2 has < 1 % octave errors in every band.

### 4. Aperiodicity

| | cat v2 | human closed v2 | human open v2 | cat v1 | human v1 |
|---|---|---|---|---|---|
| model decay rate β (median) | **0.77** | 0.39 | 0.43 | 0.254 | 0.257–0.259 |
| log10 spectral flatness, input (voiced frames) | −10.7 | −9.0 | −9.9 | −10.7 | −9.0 / −9.9 |
| log10 spectral flatness, reconstruction `zq` | **−9.7** | −8.8 | −8.9 | −7.6 | −8.6 |
| log10 spectral flatness, reconstruction oracle | −9.9 | −8.8 | −8.8 | −6.6 | −8.4 / −8.3 |

- β now differs by speaker group (cats about 2× humans). In v1 it was nearly constant (0.254–0.259), so the β head
  did not tell cats from humans.
- Cat reconstructions are much more tonal than in v1 (−9.7 vs −7.6), closer to the real cat input (−10.7), and more
  tonal than human reconstructions. v1's high-F0 frames came out as noise; that no longer happens.

### Conclusion

v1 lost the cat F0 in two places. The F0 generator mapped every cat to a human-range mean (−14.5 st) with no
contour (r = 0.04). The decoder could not render pitch above about 550 Hz. A third, hidden limit was the 880 Hz
`fo_range` cap.

**v2 removes all three:**

1. F0 now comes from the input, so the cat's pitch level and contour are carried (end to end 0.12 → 0.84).
2. The decoder renders pitch from 100 Hz to about 2 kHz, independent of cat or human conditioning.
3. `fo_range` now reaches 1760 Hz, and no frames hit it.

**What remains:**

1. **The F0 extractor above 880 Hz** (0.33 accuracy on about 10 % of cat frames, mostly very high-pitched cats).
   Training or augmenting the extractor with F0 up to about 1.5 kHz would address this.
2. **μ(z) for cats (−7.7 st).** This only matters for conversion to a cat target, where it currently has to be
   corrected with `semitone_shift` (subexp 01 v2).

**Compared with conventional VC:** in v1 the cat condition was far outside both human baselines on every F0
measure: mean −14.5 st, contour 0.04, and an F0 range beyond the decoder. In v2, cat self-reconstruction (0.84) is
close to closed-set (0.95) and open-set (0.95) humans, and its contour (0.76) is below but in the same regime as
humans (0.92–0.98). The cat-specific deficits are now a narrow high-pitch extractor problem and a biased speaker
statistic, not a structural inability to represent cat pitch.

## Limitations

- 16 cats and 16 humans with one seed. Frame counts are pooled in the band tables, so clips with more voiced frames
  weigh more.
- Ground-truth F0 for noisy cat field recordings comes from pYIN with a 200 Hz floor; residual tracking errors
  remain. Some low extractor scores on mid-pitch cats (cat02, cat07) may partly reflect pYIN errors on noisy clips.
- Self-reconstruction tests pitch, not cat timbre. A correct F0 does not mean the output sounds like a cat; subexp
  01 v2 shows the spectral gap to real cats is still mostly open.
- The sweep's top band (1600–2263 Hz) has few frames (50 cat, 150 human) and lies partly above the 1760 Hz
  `fo_range`.
- v1 and v2 differ in checkpoint, architecture and training, so per-stage comparisons are directional. The two v1
  stages (F0 generation, old decoder) do not exist in v2 in the same form.
- Spectral features beyond F0, voicing and flatness (formants, LTAS) are covered in subexp 01 and are not repeated
  here.

## Files

| file | content |
|---|---|
| `model.py` | stage-wise access to NVCNSFV2 (content / F0 extractor / semitone stats / source params / mgc → decode, streaming state, decoder-input replacement) |
| `generate.py` | clip selection, ground-truth F0, the 4 settings → `data/` |
| `generate_sweep.py` | supplemental decoder pitch sweep → `data/sweep/` |
| `analyze.py` | stage metrics, F0 bands, sweep, plots → `results/` |
| `data/manifest.csv` | clips with speaker code, `z`→`z_q` distance |
| `data/frames/*.npz` | 10 ms ground truth and stage outputs (fo, vf, β, α, μ(z), σ(z), extractor semitone) |
| `results/clip_metrics.csv`, `summary.{csv,json}`, `oracle_by_f0_band.csv`, `sweep_by_f0_band.csv` | numeric results (v2) |
| `results/stage_accuracy.png`, `f0_along_pipeline.png`, `frame_f0_histograms.png`, `decoder_pitch_sweep.png`, `contour_examples.png` | figures (v2) |
| `results_v1_old/` | results and figures of the v1 run |

## Reproduce

```bash
conda activate nvcnsf
cd experiment_01_cat_01/03.acoustic_similarity_f0_stage
python generate.py         # ~25 min: cat screening (300 clips) + 128 reconstructions
python generate_sweep.py   # ~10 min: 160 sweep syntheses
python analyze.py          # ~15 min
```
