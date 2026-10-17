"""Stage-wise access to the NVCNSF generator for subexperiment 03.

Unlike the subexperiment-01 wrapper, which only exposes end-to-end conversion,
this exposes the two stages separately so that the condition (F0 generation)
output can be inspected or replaced before the decoder (`synth`) runs:

    c    = content(x)            content encoder
    cond = condition(c, z)       F0 generation: fo = 10 * 2^((mu(z) + sigma(z) * s(c)) / 12)
    y    = synth(cond)           decoder: cyclic-noise source + neural filters

Utterances are processed in blocks of `sample_length` samples with the
streaming state carried from block to block, as in `infer.py`, so the causal
model sees the full past context (the subexperiment-01 wrapper restarted the
state at every block). A single whole-utterance call needs > 11 GB of GPU
memory in the decoder.
"""
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# appended (not prepended) so this directory's own generate.py is not shadowed by subexp 01's
sys.path.append(str(HERE.parent / '01.end_to_end_feasibility'))
import vc  # noqa: E402  (also puts the repository root on sys.path; loads TF before JAX)
from vc import SR, CHECKPOINT_DIR, load_audio, trim_silence  # noqa: E402,F401

import numpy as np  # noqa: E402

# condition outputs kept per sample (all are upsampled to the waveform rate)
COND_KEYS = ('fo', 'vf', 'beta', 'alpha', 'fo_semitone_mu', 'fo_semitone_sigma_logvar', 'fo_semitone_std_raw')


class StageModel:
    def __init__(self, model_dir=CHECKPOINT_DIR, seed=1):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint
        from hparams import get_hparams
        from models import load_checkpoint
        self.jnp = jnp

        self.hparams = hp = get_hparams()
        self.rng = {'params': jax.random.key(hp['params_seed']), 'rng_stream': jax.random.key(seed)}
        manager = orbax.checkpoint.CheckpointManager(
            str(pathlib.Path(model_dir).absolute()),
            item_names=('g_model', 'd_model', 'd_fo_model') if hp['with_d_fo'] else ('g_model', 'd_model'),
            options=orbax.checkpoint.CheckpointManagerOptions(create=False, enable_async_checkpointing=False),
        )
        self.step = manager.latest_step()
        g_state = load_checkpoint(manager, self.rng, hp)[0]
        self.params = g_state.params
        apply = g_state.apply_fn
        self.block_length = hp['sample_length']
        self.ref_length = 2 * hp['sample_length']
        # with temperature=None, z_q is the nearest of these 512 codes (SpeakerSQVAEHead)
        self.codebook = np.asarray(
            self.params['params']['speaker_encoder']['SpeakerSQVAEHead_0']['codebook'])

        self._speaker = jax.jit(lambda p, r, rng: apply(p, r, None, method='speaker_encode', rngs=rng)[1])
        # (output, stream state); the first block (state None) and later blocks compile separately
        self._content = jax.jit(lambda p, x, st, rng: apply(
            p, x, stream_data=st, method='content_encode', rngs=rng)[0::2])
        self._condition = jax.jit(lambda p, c, z, st, rng: apply(
            p, c, z, stream_data=st, method='condition', rngs=rng))
        self._synth = jax.jit(lambda p, cond, st, rng: apply(
            p, cond, stream_data=st, source_key=jax.random.key(1), noise_key=jax.random.key(2),
            method='synth', rngs=rng)[0::2])

    def speaker(self, r):
        """Returns (z, z_q, code index) for a reference waveform (tiled / center-cropped as in subexp 01)."""
        if r.size < self.ref_length:
            r = np.tile(r, math.ceil(self.ref_length / r.size))
        start = (r.size - self.ref_length) // 2
        r = self.jnp.asarray(r[start:start + self.ref_length], dtype=self.jnp.float32)[None, :, None]
        aux = self._speaker(self.params, r, self.rng)
        return aux['z'], aux['z_q'], int(np.argmax(np.asarray(aux['logits']).reshape(-1)))

    def condition(self, x, z):
        """Content encoder + condition, block-wise with streaming state.

        Returns a list of per-block cond dicts (input to `synth`)."""
        jnp = self.jnp
        n = math.ceil(x.size / self.block_length)
        xp = np.pad(x, (0, n * self.block_length - x.size)).reshape(n, self.block_length)
        conds, st_c, st_d = [], None, None
        for b in xp:
            c, st_c = self._content(self.params, jnp.asarray(b, dtype=jnp.float32)[None, :, None], st_c, self.rng)
            cond, st_d = self._condition(self.params, c, z, st_d, self.rng)
            conds.append(cond)
        return conds

    def synth(self, conds, n):
        ys, st = [], None
        for cond in conds:
            y, st = self._synth(self.params, cond, st, self.rng)
            ys.append(np.asarray(y, dtype=np.float32).reshape(-1))
        return np.concatenate(ys)[:n]

    def replace(self, conds, n, **arrays):
        """Copy of block-wise `conds` with entries replaced by per-sample arrays of length n."""
        jnp = self.jnp
        L = self.block_length
        out = [dict(c) for c in conds]
        for k, v in arrays.items():
            v = np.asarray(v, dtype=np.float32)[:n]
            v = np.pad(v, (0, len(conds) * L - v.size), mode='edge')
            for i, c in enumerate(out):
                c[k] = jnp.asarray(v[i * L:(i + 1) * L]).reshape(conds[i][k].shape)
        return out

    @staticmethod
    def numpy_cond(conds, n):
        return {k: np.concatenate([np.asarray(c[k], dtype=np.float32)[0, :, 0] for c in conds])[:n]
                for k in COND_KEYS}
