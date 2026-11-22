"""Shared helpers for subexperiment 01: model wrapper and audio loading.

The conversion calls follow `infer.py` at the repository root (NVCNSFV2:
speaker_encode -> semitone_stats, then per block content_encode / extract_fo
/ source_params_without_fo / map_fo / mgc / decode, block-wise over
`sample_length` samples with streaming state carried across blocks). A
decoder-side `semitone_shift` (as in `NVCNSFV2.convert`) can be applied after
the F0 mapping. Unlike `infer.py`, nothing is written to the current directory.
"""
import math
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same import order as `infer.py`: TensorFlow (pulled in by `preprocesses`)
# must be loaded with its GPU hidden before JAX initializes, otherwise the
# process segfaults.
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
from preprocesses.functions import remove_silence

import numpy as np
import librosa
import librosa.effects
import soundfile as sf

CHECKPOINT_DIR = pathlib.Path('/path/to/checkpoint')
SR = 24000


def load_audio(path, sr=SR):
    """Load mono audio at `sr`, peak-normalized to 0.9."""
    path = str(path)
    if path.endswith('.mp3'):
        x, orig_sr = librosa.load(path, sr=None, mono=True)
    else:
        x, orig_sr = sf.read(path, always_2d=True)
        x = x.mean(axis=1)
    x = librosa.resample(np.asarray(x, dtype=np.float64), orig_sr=orig_sr, target_sr=sr)
    peak = np.max(np.abs(x))
    return x * (0.9 / peak) if peak > 0 else x


def trim_silence(x, sr=SR, top_db=40):
    """Silence removal as done for references in `infer.py`."""
    intervals = librosa.effects.split(x, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return x
    return remove_silence(x, intervals, silence_length=sr // 8)


class VoiceConverter:
    """Thin wrapper around the NVCNSFV2 generator.

    Inputs are processed in blocks of `sample_length` samples (as in
    `infer.py`) so every jitted call sees a fixed shape. Speaker references
    (and the source clip, for its F0 statistics) are tiled / center-cropped
    to `2 * sample_length` samples.
    """

    def __init__(self, model_dir=CHECKPOINT_DIR, seed=1):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint
        from hparams import get_hparams
        from models import load_g_checkpoint
        self.jnp = jnp

        self.hparams = hp = get_hparams()
        self.rng = {
            'params': jax.random.key(hp['params_seed']),
            'rng_stream': jax.random.key(seed),
        }
        manager = orbax.checkpoint.CheckpointManager(
            str(pathlib.Path(model_dir).absolute()),
            item_names=('g_model',),
            options=orbax.checkpoint.CheckpointManagerOptions(
                create=False, enable_async_checkpointing=False),
        )
        self.step = manager.latest_step()
        if self.step is None:
            raise FileNotFoundError(f'no checkpoint found in {model_dir}')
        g_state = load_g_checkpoint(manager, self.rng, hp)
        self.params = g_state.params
        apply = g_state.apply_fn

        assert hp['sampling_rate'] == SR
        self.block_length = hp['sample_length']
        self.ref_length = 2 * hp['sample_length']

        self._speaker = jax.jit(
            lambda p, r, rng: apply(p, r, None, method='speaker_encode', rngs=rng)[1])
        self._stats = jax.jit(
            lambda p, z, rng: apply(p, z, method='semitone_stats', rngs=rng)[0])

        def block(p, x, z, stats_src, stats_tgt, shift, stream, rng):
            stream = stream or {}
            c, _, s_c = apply(p, x, stream_data=stream.get('content'),
                              method='content_encode', rngs=rng)
            fo_in, _, s_f = apply(p, x, stream_data=stream.get('fo'),
                                  method='extract_fo', rngs=rng)
            sp, _, s_s = apply(p, c, z, stream_data=stream.get('cond'),
                               method='source_params_without_fo', rngs=rng)
            fo_out = apply(p, fo_in, stats_src, stats_tgt, method='map_fo', rngs=rng)
            fo_out = fo_out * jnp.power(2.0, shift / 12.0)
            mgc, _, s_m = apply(p, c, z, stream_data=stream.get('mgc'),
                                method='mgc', rngs=rng)
            sp['fo'] = fo_out
            sp['mgc'] = mgc
            y, _, s_y = apply(p, sp, stream_data=stream.get('synth'),
                              keys={'source_key': jax.random.key(1),
                                    'noise_key': jax.random.key(2)},
                              method='decode', rngs=rng)
            return (y, fo_out, fo_in, sp['vf'],
                    {'content': s_c, 'fo': s_f, 'cond': s_s, 'mgc': s_m, 'synth': s_y})
        self._block = jax.jit(block)

    def _fit_reference(self, r):
        if r.size < self.ref_length:
            r = np.tile(r, math.ceil(self.ref_length / r.size))
        start = (r.size - self.ref_length) // 2
        return r[start:start + self.ref_length]

    def _blocks(self, x):
        n = math.ceil(x.size / self.block_length)
        return np.pad(x, (0, n * self.block_length - x.size)).reshape(n, self.block_length)

    def speaker_encode(self, r):
        """Speaker aux dict: 'z' (continuous) and 'z_q' (quantized, used for conversion)."""
        r = self.jnp.asarray(self._fit_reference(r), dtype=self.jnp.float32)[None, :, None]
        return self._speaker(self.params, r, self.rng)

    def semitone_stats(self, z):
        return self._stats(self.params, z, self.rng)

    def convert(self, x, z, z_src=None, semitone_shift=0.0):
        """Convert waveform `x` to speaker embedding `z` (quantized `z_q`).

        `z_src` is the source speaker embedding for the F0 statistics; by
        default it is encoded from `x` itself. `semitone_shift` raises the
        mapped F0 by that many semitones. Returns (y, fo, extras) at SR where
        `fo` is the F0 (Hz) fed to the decoder.
        """
        jnp = self.jnp
        if z_src is None:
            z_src = self.speaker_encode(x)['z_q']
        stats_src = self.semitone_stats(z_src)
        stats_tgt = self.semitone_stats(z)
        shift = jnp.float32(semitone_shift)
        ys, fos, fins, vfs = [], [], [], []
        stream = None
        for b in self._blocks(x):
            y, fo, fo_in, vf, stream = self._block(
                self.params, jnp.asarray(b, dtype=jnp.float32)[None, :, None],
                z, stats_src, stats_tgt, shift, stream, self.rng)
            ys.append(np.asarray(y, dtype=np.float32).reshape(-1))
            fos.append(np.asarray(fo, dtype=np.float32).reshape(-1))
            fins.append(np.asarray(fo_in, dtype=np.float32).reshape(-1))
            vfs.append(np.asarray(vf, dtype=np.float32).reshape(-1))
        n = x.size
        return (np.concatenate(ys)[:n], np.concatenate(fos)[:n],
                {'fo_in': np.concatenate(fins)[:n], 'vf': np.concatenate(vfs)[:n]})
