import os
import pytest

pytestmark = [pytest.mark.accelerator,pytest.mark.skipif(os.getenv('FDR_TEST_DEVICE')!='cuda',reason='CUDA model smoke tests are opt-in')]

@pytest.mark.parametrize('op',['delta','gdn2'])
def test_fla_model_layer_uses_backend(op):
    import torch
    import fast_delta_rule as fdr
    # FLA's package initializer may import layers before enable().
    from fla.layers.delta_net import DeltaNet
    from fla.layers.gdn2 import GatedDeltaNet2
    model=(DeltaNet(hidden_size=256,num_heads=2,layer_idx=0) if op=='delta' else
           GatedDeltaNet2(hidden_size=256,num_heads=2,head_dim=128,layer_idx=0)).cuda().train()
    x=torch.randn(1,128,256,device='cuda',requires_grad=True)
    fdr.enable(ops=(op,));fdr.stats(reset=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,use_cache=False)[0]
    out.float().square().mean().backward()
    fdr.assert_used((op,))
    assert torch.isfinite(x.grad).all()
    fdr.disable()

def test_qwen_adapter_trains_a_small_model():
    import torch
    import fast_delta_rule as fdr
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    cfg=Qwen3_5TextConfig(hidden_size=256,intermediate_size=512,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=128,vocab_size=256,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=128,
        linear_value_head_dim=128,layer_types=['linear_attention'],use_cache=False)
    cfg._attn_implementation='sdpa'
    model=Qwen3_5ForCausalLM(cfg).cuda().train()
    tokens=torch.randint(0,256,(1,128),device='cuda')
    fdr.enable(ops=('gdn',),integrations=('qwen3_5',));fdr.stats(reset=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=tokens,labels=tokens).loss
    loss.backward()
    fdr.assert_used(('gdn',))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    fdr.disable()
