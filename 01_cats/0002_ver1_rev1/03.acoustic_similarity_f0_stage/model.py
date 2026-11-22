"""Stage-wise access to the NVCNSFV2 generator for subexperiment 03.

Exposes the stages separately so that the F0 stage output can be inspected or
replaced before the decoder runs (V2 pipeline, as in `infer.py`):

    c      = content(x)                      content encoder
    fo_in  = extract_fo(x)                   F0 extractor (reads the input waveform)
    mu,sig = semitone_stats(z)               speaker semitone statistics
    fo     = map_fo(fo_in, stats(z_src), stats(z_tgt))   (identity for self-reconstruction)
    p      = source_params_without_fo(c, z)  vf, alpha, beta; mgc = mgc(c, z)
    y      = decode(p | {fo, mgc})           decoder: cyclic-noise source + neural filters

In contrast to the pre-V2 model, F0 is no longer generated from (c, z): the
contour comes from the input, and z only supplies the statistics used to map
it to the target speaker (plus a decoder-side `semitone_shift`).

Utterances are processed in blocks of `sample_length` samples with the
streaming state carried from block to block, as in `infer.py`. A single
whole-utterance call needs > 11 GB of GPU memory in the decoder.
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

# condition outputs kept per sample (all are upsampled to the waveform rate).
# fo_semitone_* are the speaker semitone statistics mu(z), sigma(z) (sigma as log-variance).
COND_KEYS = ('fo', 'vf', 'beta', 'alpha', 'fo_semitone_mu', 'fo_semitone_sigma_logvar', 'fo_semitone_std_raw')
EXTRA = ('fo_semitone_mu', 'fo_semitone_sigma_logvar', 'fo_semitone_std_raw')  # not decoder inputs


class StageModel:
    def __init__(self, model_dir=CHECKPOINT_DIR, seed=1):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint
        from hparams import get_hparams
        from models import load_g_checkpoint
        self.jnp = jnp

        self.hparams = hp = get_hparams()
        self.rng = {'params': jax.random.key(hp['params_seed']), 'rng_stream': jax.random.key(seed)}
        manager = orbax.checkpoint.CheckpointManager(
            str(pathlib.Path(model_dir).absolute()),
            item_names=('g_model',),
            options=orbax.checkpoint.CheckpointManagerOptions(create=False, enable_async_checkpointing=False),
        )
        self.step = manager.latest_step()
        g_state = load_g_checkpoint(manager, self.rng, hp)
        self.params = g_state.params
        apply = g_state.apply_fn
        self.block_length = hp['sample_length']
        self.ref_length = 2 * hp['sample_length']
        # with temperature=None, z_q is the nearest of these codes (SpeakerSQVAEHead)
        self.codebook = np.asarray(
            self.params['params']['speaker_encoder']['SpeakerSQVAEHead_0']['codebook'])
        L = self.block_length

        self._speaker = jax.jit(lambda p, r, rng: apply(p, r, None, method='speaker_encode', rngs=rng)[1])
        self._stats = jax.jit(lambda p, z, rng: apply(p, z, method='semitone_stats', rngs=rng)[0])

        def cond_block(p, x, z, stats, st, rng):
            st = st or {}
            c, _, s_c = apply(p, x, stream_data=st.get('content'), method='content_encode', rngs=rng)
            fo_in, _, s_f = apply(p, x, stream_data=st.get('fo'), method='extract_fo', rngs=rng)
            sp, _, s_s = apply(p, c, z, stream_data=st.get('cond'), method='source_params_without_fo', rngs=rng)
            fo = apply(p, fo_in, stats, stats, method='map_fo', rngs=rng)  # self-mapping
            mgc, _, s_m = apply(p, c, z, stream_data=st.get('mgc'), method='mgc', rngs=rng)
            sp = dict(sp)
            sp['fo'], sp['mgc'] = fo, mgc
            sp['fo_semitone_mu'] = jnp.broadcast_to(jnp.repeat(stats['mu'], L // stats['mu'].shape[1], axis=1), fo.shape)
            sp['fo_semitone_sigma_logvar'] = jnp.broadcast_to(
                jnp.repeat(2 * jnp.log(stats['sigma']), L // stats['sigma'].shape[1], axis=1), fo.shape)
            sp['fo_semitone_std_raw'] = 12. * jnp.log2(fo_in / 10.)  # extractor semitone before mapping
            return sp, {'content': s_c, 'fo': s_f, 'cond': s_s, 'mgc': s_m}
        self._cond = jax.jit(cond_block)

        def synth_block(p, cond, st, rng):
            cond = {k: v for k, v in cond.items() if k not in EXTRA}
            y, _, s = apply(p, cond, stream_data=st,
                            keys={'source_key': jax.random.key(1), 'noise_key': jax.random.key(2)},
                            method='decode', rngs=rng)
            return y, s
        self._synth = jax.jit(synth_block)

    def speaker(self, r):
        """Returns (z, z_q, code index) for a reference waveform (tiled / center-cropped as in subexp 01)."""
        if r.size < self.ref_length:
            r = np.tile(r, math.ceil(self.ref_length / r.size))
        start = (r.size - self.ref_length) // 2
        r = self.jnp.asarray(r[start:start + self.ref_length], dtype=self.jnp.float32)[None, :, None]
        aux = self._speaker(self.params, r, self.rng)
        return aux['z'], aux['z_q'], int(np.argmax(np.asarray(aux['logits']).reshape(-1)))

    def condition(self, x, z):
        """Self-reconstruction parameters (F0 extractor + source params + mgc), block-wise with streaming state.

        Returns a list of per-block cond dicts (input to `synth`)."""
        jnp = self.jnp
        n = math.ceil(x.size / self.block_length)
        xp = np.pad(x, (0, n * self.block_length - x.size)).reshape(n, self.block_length)
        stats = self._stats(self.params, z, self.rng)
        conds, st = [], None
        for b in xp:
            cond, st = self._cond(self.params, jnp.asarray(b, dtype=jnp.float32)[None, :, None], z, stats, st, self.rng)
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
