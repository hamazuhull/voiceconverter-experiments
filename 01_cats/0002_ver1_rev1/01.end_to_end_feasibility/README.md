# 01. End-to-end feasibility test

## Objective

Check that the pretrained NVCNSF model can run human → cat conversion end to end, and measure how close the
generated "cat-like" sounds get to the target cat acoustically. The comparison point is conventional human → human
VC with the same model and the same sources:

| condition | target reference | why it is a reference point |
|---|---|---|
| `human2cat` | real cat vocalization (meow-10k) | the condition under study |
| `open_intra` / `open_inter` | unseen Common Voice speaker, same / other gender | open-set VC, the realistic deployment case (cats are open-set too) |
| `closed_intra` / `closed_inter` | jvs training speaker, same / other gender | closed-set VC, upper bound of what the model does well |

Concrete questions:
1. Does the pipeline produce valid, finite audio for cat references? (feasibility)
2. How much of the source→target acoustic distance does a conversion close, and how does human→cat compare with
   the conventional settings?
3. Does the output pitch follow the target into the cat F0 range?

### Model versions

| | v1 (previous run) | **v2 (this report)** |
|---|---|---|
| model | NVCNSF, `~/Documents/Repos/nvcnsf_app/app_py/checkpoints/`, step 500000 | **NVCNSFV2**, `/path/to/checkpoint/`, step 100000 |
| F0 path | F0 generated from content features + speaker code, `fo_range` 27.5–880 Hz | F0 **extracted from the input**, mapped to the target with speaker semitone statistics μ(z), σ(z), `fo_range` 27.5–1760 Hz |
| pitch control | none | decoder-side `semitone_shift` |
| human→cat F0 | whatever the model produced | **per-pair shift matched to the cat reference** (see Method) |

The v1 outputs are kept in `results_v1_old/` (summary, metrics, figures, manifest). Every v1 number in this report
comes from there.

## Data

Unchanged from v1 (same seed, same selection):

| role | source | selection (seed 0) |
|---|---|---|
| sources | Common Voice ja `test.tsv` | 4 male + 4 female speakers (`gender` labeled), one 3–7 s utterance each |
| open-set refs | Common Voice ja `test.tsv` | 2 M + 2 F speakers disjoint from the sources; up to 6 clips concatenated (≥ 4 s after silence removal) |
| closed-set refs | jvs `nonpara30` | jvs001 (M), jvs003 (M), jvs002 (F), jvs004 (F); 3 utterances concatenated |
| cat refs | meow-10k `audios/` (16 kHz) | 120 random clips screened with pYIN (150–2000 Hz); top 6 by voiced ratio with median F0 in 250–1500 Hz |

The cat screening step exists because many meow-10k clips are behaviour recordings, not vocalizations. A first
ranking by voiced ratio alone picked clips whose "F0" was pinned at the search floor (purrs / low-frequency
background noise detected as voiced). The 250–1500 Hz median-F0 window removes those. Screening results:
`data/cat_candidates.csv`.

Selected cats: `cat00` fs_cat_class_00319, `cat01` tz_cat_class_00415, `cat02` fs_cat_class_00201,
`cat03` tz_cat_class_00419, `cat04` tz_cat_class_00204, `cat05` fs_cat_class_00705.

Total: 8 sources × 14 references = **112 conversions** (48 human→cat, 16 per baseline condition).

## Method

`generate.py` runs the `infer.py` pipeline through the wrapper in `vc.py`. For each block of `sample_length` samples,
the streaming state is carried across blocks and the steps are:
1. Encode the content, extract F0 from the input, and predict the source parameters (vf, α, β).
2. Map F0 from the source's semitone statistics to the target's (`map_fo`), then apply `semitone_shift`.
3. Compute the mgc and decode.

References are silence-trimmed as in `infer.py`, then tiled / center-cropped to 2 × `sample_length` (2.73 s). The
source utterance is processed the same way to obtain its own statistics. The quantized speaker code `z_q` drives the
conversion.

**Cat-matched pitch shift (human→cat only).** Each pair is first converted with shift 0; that output is kept in
`data/conv_s0/`. The shift is then `12·log2(F0_cat / F0_out)`, where both F0s are pYIN medians (150–2000 Hz, the
cat-screening settings), clipped to ±24 st. The conversion is repeated with that shift and saved to `data/conv/`.
The shift and target are logged in `data/manifest.csv` (`shift_st`, `target_f0_hz`). Human baselines use shift 0,
so they measure the model's own F0 mapping.

`analyze.py` computes, on 16 kHz versions of every file (common bandwidth: the cat recordings are 16 kHz):

| metric | definition |
|---|---|
| `mfcc` | Euclidean distance of utterance-mean MFCC c1–c24 over non-silent frames (librosa units) |
| `ltas` | RMS difference (dB) of level-normalized long-term average spectra, 100 Hz–7 kHz |
| `spk` | cosine similarity of the model's own continuous speaker embedding `z` |
| `f0` | pYIN median F0 of voiced frames (60–1600 Hz); cat targets use the 150–2000 Hz estimate (see limitations) |
| `gap_closed` | `1 − d(conv, ref) / d(src, ref)`: 1 = reached the target, 0 = did not move, < 0 = moved away |

To give the distances a scale, `reference_spread.csv` contains the distances among the real references
themselves (cat–cat, human–human, cat–human).

## Results

### Main table (v2; means over conversions)

`results/summary.csv` has ± std and `results/metrics.csv` has every conversion.

| condition | MFCC d(src,ref) | MFCC d(conv,ref) | MFCC gap closed | LTAS gap closed | spk cos (conv,ref) | target F0 [Hz] | output F0 [Hz] | \|F0 error\| [st] | voiced ratio (out / ref) |
|---|---|---|---|---|---|---|---|---|---|
| **human→cat** | 62.8 | 47.1 | **0.23** | **0.16** | 0.54 | 546 | 527 | **1.3** | 0.64 / 0.75 |
| open intra | 37.6 | 31.3 | 0.16 | 0.10 | 0.49 | 158 | 146 | 5.1 | 0.56 / 0.45 |
| open inter | 39.9 | 30.2 | 0.23 | 0.16 | 0.41 | 158 | 137 | 5.7 | 0.57 / 0.45 |
| closed intra | 38.9 | 30.1 | 0.22 | 0.09 | 0.59 | 191 | 184 | 4.5 | 0.57 / 0.74 |
| closed inter | 45.1 | 31.6 | 0.29 | 0.13 | 0.26 | 191 | 169 | 5.2 | 0.57 / 0.74 |

Spread among the real references (mean pairwise MFCC distance): cat–cat 58.3, cat–human 55.3, human–human 33.0.
These depend only on the audio, so they are the same in v1 and v2.

### Change from v1

| condition | MFCC gap closed v1 → v2 | LTAS gap closed v1 → v2 | \|F0 error\| [st] v1 → v2 | output F0 [Hz] v1 → v2 | spk cos v1 → v2 |
|---|---|---|---|---|---|
| **human→cat** | 0.15 → **0.23** | 0.10 → **0.16** | 15.9 → **1.3** | 228 → **527** (target 546) | 0.37 → 0.54 |
| open intra | 0.22 → 0.16 | 0.20 → 0.10 | 6.5 → 5.1 | 169 → 146 | 0.58 → 0.49 |
| open inter | 0.28 → 0.23 | 0.23 → 0.16 | 5.9 → 5.7 | 202 → 137 | 0.52 → 0.41 |
| closed intra | 0.31 → 0.22 | 0.17 → 0.09 | 3.2 → 4.5 | 218 → 184 | 0.75 → 0.59 |
| closed inter | 0.39 → 0.29 | 0.22 → 0.13 | 2.2 → 5.2 | 210 → 169 | 0.66 → 0.26 |

Human→cat is the only condition that improved on every metric. All human baselines moved less toward their target
than in v1. Speaker-embedding cosines come from each model's own encoder, so they are compared only within a version.

### Per cat (v2, human→cat; 8 sources each)

| cat | target F0 [Hz] | output F0 median [Hz] | \|F0 error\| [st] | applied shift [st] | MFCC gap closed |
|---|---|---|---|---|---|
| cat00 | 1228 | 1220 | 0.3 | +11.1 … +24.0 | **0.53** |
| cat01 | 348 | 272 | **5.1** | +6.9 … +14.4 | 0.07 |
| cat02 | 382 | 385 | 0.2 | −9.1 … +3.9 | 0.22 |
| cat03 | 566 | 564 | 0.1 | −2.5 … +10.6 | 0.07 |
| cat04 | 473 | 472 | 1.3 | +8.3 … +20.0 | 0.30 |
| cat05 | 280 | 287 | 0.6 | −4.5 … +10.6 | 0.17 |

Figures:
- `results/distances_by_condition.png`: before/after distances per condition
- `results/gap_closed.png`: fraction of the gap closed (main comparison)
- `results/f0_target_vs_output.png`: target vs output median F0
- `results/ltas.png`: average spectra of sources, cat references, and outputs
- `results/spectrogram_examples.png`: one source, its best human→cat output (by MFCC distance) with the cat
  target, and an open intra-gender output

### Findings

1. **The pipeline runs end to end with the new model.** All 112 conversions finished with finite output and the
   expected length, as in v1.
2. **Pitch now reaches the cat range (v1's main failure is fixed).** Output F0 follows the cat target from 280 to
   1230 Hz with a mean error of 1.3 st (median 0.2 st). In v1 the output stayed at 65–400 Hz with a 16 st error.
   cat00 (≈ 1.2 kHz) was impossible in v1 because of the 880 Hz `fo_range` cap. It is now reproduced at 1220 Hz and
   is also the cat with the largest spectral gain (0.53). Two things made this possible: the decoder renders F0 up
   to ≥ 1.9 kHz (subexp 03) and the semitone shift moves the F0 there. Without the shift, μ(z) for cats is too low
   (subexp 03), so the shift is necessary. 43 of 48 pairs end within 3 st of the target. The exceptions are mostly
   cat01: its median-based shift undershoots because the cat01 reference contains low non-meow components, so its
   output stays at 166–351 Hz.
3. **Human→cat now moves toward the target about as much as conventional VC, but the remaining gap is large.**
   - Human→cat closes 23 % of the MFCC gap (v1 15 %) and 16 % of the LTAS gap (v1 10 %). This is within the range
     of the v2 human baselines (16–29 % MFCC, 9–16 % LTAS). In v1 it was below every baseline.
   - In absolute terms the output is still 47.1 from its cat target. That is below the cat–human spread (55.3) but
     far from the 30–32 the human conditions reach. Because the starting gap is about 1.5× larger (62.8 vs 38–45),
     an equal fraction leaves much more distance.
   - The outputs also moved away from the source: d(conv, src) went from 34.0 to 51.5.
4. **The spectra show what pitch changed and what it did not** (`ltas.png`, `spectrogram_examples.png`).
   - Changed: the output LTAS lost the human low-frequency (< 300 Hz) peak, gained a bump around 1–1.5 kHz, and
     follows the cats' falling slope below 500 Hz.
   - Not changed: it still lacks most of the cats' 0.8–3 kHz energy. In the spectrogram the cat00 output has a
     sustained ≈ 1.2 kHz harmonic series like the cat, but it keeps the source's syllable timing and its broadband
     consonant noise up to 8 kHz. The cat's long, smooth tonal calls with gliding pitch are not produced.
   - The outputs are human utterances at cat pitch, not cat calls.
5. **Voicing moved toward the cat.** The output voiced ratio is 0.64 (v1 0.44; cat references 0.75).
6. **The model's own speaker embedding** rates the outputs closer to the cat (cosine 0.54, v1 0.37). That is still
   below the v2 cat–cat similarity (0.72). Because the encoder that judges the output is the one that conditioned it,
   this is optimistic and is not taken as evidence of cat-likeness. Subexperiment 2 tests this with an independent
   classifier.

**Compared with conventional VC:**
- In v1, the closed-set inter-gender condition closed 39 % of the MFCC gap with a 2 st F0 error, while human→cat
  closed 15 % with a 16 st error, so the species gap looked out of reach.
- In v2, with pitch matched, human→cat (23 %) is level with open-set inter-gender (23 %) and closed-set intra-gender
  (22 %) conversion. Its remaining distance (47 vs 30–32) is still far larger, and the residual is spectral and
  temporal (timbre, call shape), not pitch.
- The human baselines themselves got worse in v2 (gap closed −0.05 to −0.10, F0 error 4.5–5.7 st). So part of the
  "catching up" is the baselines dropping, not only human→cat improving. Open-set is still worse than closed-set
  (0.16–0.23 vs 0.22–0.29).

## Limitations

- Small sample (8 sources, 6 cats), single seed. This is a feasibility check, not a statistical comparison; no
  significance tests are reported.
- **The pitch shift is oracle-informed.** It is computed from the target cat's measured F0, so v2 human→cat is an
  upper bound for pitch control, not a fully automatic conversion. The shift-0 outputs are saved in `data/conv_s0/`,
  but their metrics have not been computed. The contributions of the new model and of the shift are therefore not
  separated.
- v1 and v2 differ in checkpoint, architecture and training. Changes in the human baselines show that absolute
  numbers are not strictly comparable. The comparisons above are within-version (human→cat vs baselines) or
  directional.
- Cat clips are 16 kHz field recordings with background noise. For one cat (cat05), the wide-range pYIN locks onto
  low-frequency background at 157 Hz (vs 280 Hz in the 150–2000 Hz band). This is why cat target F0 uses the
  meow-band estimate. Both values are kept in `metrics.csv` (`f0_ref`, `f0_ref_meow`).
- Cat clips can mix meows with other sounds (purr, noise), and the 2.73 s reference crop may include them. This is
  the likely cause of cat01's poor shift.
- Utterance-mean MFCC and LTAS ignore temporal structure (a cat call's pitch contour, duration), which
  subexperiment 3 covers.
- The speaker-embedding cosine is noisy (values spanning −0.8 to 1), so it is reported only as a raw similarity.

## Files

| file | content |
|---|---|
| `vc.py` | NVCNSFV2 model wrapper (follows `infer.py`; `convert(x, z, z_src, semitone_shift)`), audio loading, silence trimming |
| `generate.py` | data selection, cat screening, cat-matched shift, conversion → `data/` |
| `analyze.py` | metrics, summaries, plots → `results/` |
| `data/manifest.csv` | one row per conversion (paths, condition, speakers, `shift_st`, `target_f0_hz`) |
| `data/{src,ref,conv}/*.wav` | audio (24 kHz); `data/conv_s0/*.wav`: human→cat at shift 0; `data/conv_fo/*.npy`: decoder F0 at 100 frames/s |
| `results/metrics.csv`, `summary.{csv,json}`, `reference_spread.csv` | numeric results (v2) |
| `results_v1_old/` | summary, metrics, figures and manifest of the v1 run |
| `generate.log`, `analyze.log` | logs of the last run |

## Reproduce

```bash
conda activate nvcnsf
cd experiment_01_cat_01/01.end_to_end_feasibility
python generate.py   # ~30 min (cat screening + 112 conversions, plus 48 shift-0 passes)
python analyze.py    # ~5 min
```

Options: `generate.py --n-src-per-gender 4 --n-ref-per-gender 2 --n-cat 6 --n-cat-pool 120 --seed 0`.
Note: `vc.py` imports TensorFlow (with its GPU hidden) before JAX, in the same order as `infer.py`. Importing
`preprocesses` after JAX has started causes a segfault.
