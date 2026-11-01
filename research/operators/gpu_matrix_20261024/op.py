"""Backward-only patch contexts. Public forward and public save flag are unchanged."""
import contextlib
import importlib
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ['KDA_DEVICE'] = 'cuda'
os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
sys.path.insert(0, str(ROOT))
import run_cuda as e
from kernels.kda_residual_local_beta import LocalBetaResidual
from kernels.kda_intra_beta_masked import chunk_kda_bwd_intra as residual_intra
from kernels.kda_fused_tile import fused_wy, fused_intra
from matrix_intra import MatrixIntra

ORIGINAL_INTRA = e.bwd.chunk_kda_bwd_intra
ORIGINAL_WY = e.bwd.chunk_kda_bwd_wy_dqkg_fused
ORIGINAL_KERNEL = e.bwd.chunk_kda_bwd_kernel_wy_dqkg_fused
ORIGINAL_CUMSUM = e.bwd.chunk_local_cumsum
chunkmod = importlib.import_module('fla.ops.kda.chunk')
ORIGINAL_FORWARD = chunkmod.chunk_kda_fwd
VARIANTS = ('original', 'residual', 'matrix', 'combined', 'previous_fused')


@contextlib.contextmanager
def implementation(variant, bk=32, warps=4, lowp=True):
    assert variant in VARIANTS
    attrs = ('chunk_kda_bwd_intra', 'chunk_kda_bwd_wy_dqkg_fused',
             'chunk_kda_bwd_kernel_wy_dqkg_fused', 'chunk_local_cumsum')
    saved = {a: getattr(e.bwd, a) for a in attrs}
    env = {a: os.environ.get(a) for a in ('KDA_TILE_BK', 'KDA_INTRA_DOT', 'KDA_COMPENSATE', 'KDA_FUSED_LOGIT_SCALE')}
    matrix = MatrixIntra(ORIGINAL_INTRA, residual_intra, variant == 'combined', bk, warps, lowp)
    try:
        e.bwd.chunk_kda_bwd_intra = ORIGINAL_INTRA
        e.bwd.chunk_kda_bwd_wy_dqkg_fused = ORIGINAL_WY
        e.bwd.chunk_kda_bwd_kernel_wy_dqkg_fused = ORIGINAL_KERNEL
        e.bwd.chunk_local_cumsum = ORIGINAL_CUMSUM
        if variant in ('residual', 'combined'):
            e.bwd.chunk_kda_bwd_kernel_wy_dqkg_fused = LocalBetaResidual()
            e.bwd.chunk_kda_bwd_intra = residual_intra
        if variant in ('matrix', 'combined'):
            e.bwd.chunk_kda_bwd_intra = matrix
        if variant == 'previous_fused':
            os.environ.update(KDA_TILE_BK='32', KDA_INTRA_DOT='lowp', KDA_COMPENSATE='0', KDA_FUSED_LOGIT_SCALE='0')
            e.bwd.chunk_kda_bwd_wy_dqkg_fused = fused_wy
            e.bwd.chunk_kda_bwd_intra = fused_intra
            e.bwd.chunk_local_cumsum = lambda g, **kw: g
        assert chunkmod.chunk_kda_fwd is ORIGINAL_FORWARD
        yield matrix
        assert chunkmod.chunk_kda_fwd is ORIGINAL_FORWARD
    finally:
        for a, v in saved.items():
            setattr(e.bwd, a, v)
        for a, v in env.items():
            if v is None:
                os.environ.pop(a, None)
            else:
                os.environ[a] = v
