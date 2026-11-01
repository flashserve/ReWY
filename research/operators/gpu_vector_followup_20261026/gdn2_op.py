"""Factorial ablation against the current SPLIT GDN2 recommendation."""
from gdn2_base import GDN2 as Base, c
from gdn2_kernels_gpu import residual, matrix_intra
from vector_state_gpu import state_bwd

class GDN2(Base):
    variants = ['original', 'incumbent', 'state', 'consumer', 'combined']

    def backward(self, xs, dys, fw, variant, svf=False):
        assert variant in self.variants
        attrs = ('chunk_gdn2_bwd_intra', 'chunk_gdn2_bwd_wy_dqkg_fused', 'chunk_gated_delta_rule_bwd_dhu')
        saved = {a: getattr(c, a) for a in attrs}
        try:
            if variant != 'original':
                c.chunk_gdn2_bwd_wy_dqkg_fused = residual
                c.chunk_gdn2_bwd_intra = matrix_intra
            if variant in ('combined',):
                c.chunk_gated_delta_rule_bwd_dhu = state_bwd
            if variant in ('combined',):
                from gdn2_consumer_gpu import residual as producer, matrix_intra as consumer
                c.chunk_gdn2_bwd_wy_dqkg_fused = producer
                c.chunk_gdn2_bwd_intra = consumer
            return super().backward(xs, dys, fw, 'original', svf)
        finally:
            for a, v in saved.items():
                setattr(c, a, v)
