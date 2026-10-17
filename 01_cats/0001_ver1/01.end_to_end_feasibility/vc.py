"""Shared helpers for subexperiment 01: model wrapper and audio loading.

The conversion calls follow `infer.py` at the repository root
(speaker_encode -> content_encode -> condition -> synth, block-wise over
`sample_length` samples). Unlike `infer.py`, nothing is written to the
current directory.
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
from preprocesses import remove_silence

import numpy as np
import librosa
import librosa.effects
import soundfile as sf

CHECKPOINT_DIR = pathlib.Path('/path/to/checkpoints').expanduser()
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
    """Thin wrapper around the NVCNSF generator.

    Inputs are processed in blocks of `sample_length` samples (as in
    `infer.py`) so every jitted call sees a fixed shape. Speaker references
    are tiled / center-cropped to `2 * sample_length` samples.
    """

    def __init__(self, model_dir=CHECKPOINT_DIR, seed=1):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint
        from hparams import get_hparams
        from models import load_checkpoint
        self.jnp = jnp

        self.hparams = hp = get_hparams()
        self.rng = {
            'params': jax.random.key(hp['params_seed']),
            'rng_stream': jax.random.key(seed),
        }
        manager = orbax.checkpoint.CheckpointManager(
            str(pathlib.Path(model_dir).absolute()),
            item_names=('g_model', 'd_model', 'd_fo_model') if hp['with_d_fo'] else ('g_model', 'd_model'),
            options=orbax.checkpoint.CheckpointManagerOptions(
                create=False, enable_async_checkpointing=False),
        )
        self.step = manager.latest_step()
        if self.step is None:
            raise FileNotFoundError(f'no checkpoint found in {model_dir}')
        g_state = load_checkpoint(manager, self.rng, hp)[0]
        self.params = g_state.params
        apply = g_state.apply_fn

        assert hp['sampling_rate'] == SR
        self.block_length = hp['sample_length']
        self.ref_length = 2 * hp['sample_length']

        self._speaker = jax.jit(
            lambda p, r, rng: apply(p, r, None, method='speaker_encode', rngs=rng)[1])
        self._content = jax.jit(
            lambda p, x, rng: apply(p, x, method='content_encode', rngs=rng)[0])
        self._condition = jax.jit(
            lambda p, c, z, rng: apply(p, c, z, method='condition', rngs=rng)[0])
        self._synth = jax.jit(
            lambda p, cond, rng: apply(
                p, cond,
                source_key=jax.random.key(1),
                noise_key=jax.random.key(2),
                method='synth', rngs=rng)[0])

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

    def convert(self, x, z):
        """Convert waveform `x` to speaker embedding `z`. Returns (y, fo) at SR."""
        jnp = self.jnp
        ys, fos = [], []
        for b in self._blocks(x):
            c = self._content(self.params, jnp.asarray(b, dtype=jnp.float32)[None, :, None], self.rng)
            cond = self._condition(self.params, c, z, self.rng)
            ys.append(np.asarray(self._synth(self.params, cond, self.rng), dtype=np.float32).reshape(-1))
            fos.append(np.asarray(cond['fo'], dtype=np.float32).reshape(-1))
        return np.concatenate(ys)[:x.size], np.concatenate(fos)[:x.size]
