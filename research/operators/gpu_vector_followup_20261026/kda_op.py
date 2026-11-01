"""Factorial ablation against the current JOINT KDA recommendation."""
from common import *
from kda_base import KDA as Base, c
from kernels.kda_fused_tile import fused_wy, fused_intra
from vector_state_gpu import state_bwd

class KDA(Base):
    variants = ['original', 'incumbent', 'state', 'consumer', 'combined']

    def backward(self, xs, dys, fw, variant, svf=False):
        assert variant in self.variants
        attrs = ('chunk_kda_bwd_intra', 'chunk_kda_bwd_wy_dqkg_fused', 'chunk_local_cumsum', 'chunk_gated_delta_rule_bwd_dhu')
        saved = {a: getattr(c, a) for a in attrs}
        env = {a: os.environ.get(a) for a in ('KDA_TILE_BK', 'KDA_INTRA_DOT', 'KDA_COMPENSATE', 'KDA_FUSED_LOGIT_SCALE')}
        try:
            if variant != 'original':
                os.environ.update(KDA_TILE_BK='32', KDA_INTRA_DOT='lowp', KDA_COMPENSATE='0', KDA_FUSED_LOGIT_SCALE='0')
                c.chunk_kda_bwd_wy_dqkg_fused = fused_wy
                c.chunk_kda_bwd_intra = fused_intra
                c.chunk_local_cumsum = lambda g, **kw: g
            if variant in ('combined',):
                c.chunk_gated_delta_rule_bwd_dhu = state_bwd
            if variant in ('combined',):
                from kda_consumer_gpu import fused_intra as consumer
                c.chunk_kda_bwd_intra = consumer
            return super().backward(xs, dys, fw, 'original', svf)
        finally:
            for a, v in saved.items():
                setattr(c, a, v)
            for a, v in env.items():
                if v is None:
                    os.environ.pop(a, None)
                else:
                    os.environ[a] = v
