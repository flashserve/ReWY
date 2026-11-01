#include "acl/acl.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/matrix/matmul_tiling.h"
#include "residual_config.h"
#include <algorithm>
#include <iostream>

extern "C" void launch_kda_residual_native(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, ResidualConfig);
extern "C" void launch_kda_matmul_probe(void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, ResidualConfig);
extern "C" void launch_kda_intra_native(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    uint8_t*,uint8_t*,uint8_t*,IntraConfig);

namespace {
AscendC::tiling::TCubeTiling MakeTiling(int m, int n, int k, bool ta, bool tb) {
    using namespace matmul_tiling;
    PlatformInfo platform;
    platform.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    platform.l1Size = 512 * 1024;
    platform.l0ASize = 64 * 1024;
    platform.l0BSize = 64 * 1024;
    platform.l0CSize = 128 * 1024;
    platform.ubSize = 192 * 1024;
    MatmulApiTiling tiling(platform);
    tiling.SetAType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_BF16, ta);
    tiling.SetBType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_BF16, tb);
    tiling.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_FLOAT);
    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(false);
    tiling.SetFixSplit(std::min(m,32), std::min(n,64), 64);
    tiling.SetDoubleBuffer(true, true, false, false);
    tiling.SetBufferSpace(128 * 1024, 32 * 1024, 32 * 1024);
    AscendC::tiling::TCubeTiling out;
    TORCH_CHECK(tiling.GetTiling(out) == 0, "Matmul tiling generation failed");
    SysTilingTempBufSize bytes;
    TORCH_CHECK(MatmulGetTmpBufSizeV2(out, bytes) == 0, "Matmul buffer query failed");
    std::cerr << "residual Matmul M=" << m << " N=" << n << " K=" << k
              << " UB=" << bytes.ubSize << " L1=" << bytes.l1Size
              << " L0C=" << bytes.l0cSize << " depthA1=" << out.depthA1
              << " depthB1=" << out.depthB1 << std::endl;
    std::cerr << "Matmul sharing mode=" << out.shareMode << " L1=" << out.shareL1Size
              << " L0C=" << out.shareL0CSize << " UB=" << out.shareUbSize << std::endl;
    TORCH_CHECK(bytes.ubSize <= 32 * 1024, "Matmul implicit UB exceeds explicit budget");
    return out;
}

void Producer(const at::Tensor& value, const at::Tensor& residual,
    const at::Tensor& inverse, const at::Tensor& dr, const at::Tensor& state,
    const at::Tensor& key, const at::Tensor& gate, const at::Tensor& beta,
    at::Tensor dk, at::Tensor dg, at::Tensor dv, at::Tensor db, at::Tensor dm,
    const at::Tensor& workspace) {
    for (const auto* t : {&value, &residual, &inverse, &dr, &key, &beta, &state}) {
        TORCH_CHECK(t->is_privateuseone() && t->scalar_type() == at::kBFloat16 && t->is_contiguous(),
                    "native residual expects contiguous BF16 input on NPU");
    }
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&gate, &dk, &dg, &db, &dm}) {
        TORCH_CHECK(t->is_privateuseone() && t->scalar_type() == at::kFloat && t->is_contiguous(),
                    "native residual expects contiguous FP32 state/gate/gradient");
    }
    TORCH_CHECK(value.dim() == 4 && value.size(1) % 64 == 0 && value.size(3) == 128,
                "native residual requires [B,T,H,128], T divisible by 64");
    TORCH_CHECK(key.size(3) == 128 && inverse.size(3) == 64, "unsupported native shape");
    TORCH_CHECK(workspace.is_privateuseone() && workspace.scalar_type() == at::kByte &&
                workspace.numel() >= 16 * 1024 * 1024, "insufficient Matmul system workspace");
    static const auto pTiling = MakeTiling(32, 128, 64, true, false);
    static const auto cTiling = MakeTiling(32, 64, 128, false, true);
    ResidualConfig cfg{pTiling, cTiling, uint32_t(value.size(0)), uint32_t(value.size(1)),
        uint32_t(value.size(2)), uint32_t(key.size(2)),
        uint32_t(value.size(0) * value.size(1) / 64 * value.size(2) * 2)};
    // stream(true)先排空框架host task queue，保证直调与既有Triton kernel顺序一致。
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    const uint32_t blocks = std::min<uint32_t>(24, (cfg.tasks + 1) / 2);
    auto ptr = [](const at::Tensor& x) { return reinterpret_cast<uint8_t*>(x.data_ptr()); };
    launch_kda_residual_native(blocks, stream, ptr(value), ptr(residual), ptr(inverse), ptr(dr),
        ptr(state), ptr(key), ptr(gate), ptr(beta), ptr(dk), ptr(dg), ptr(dv), ptr(db), ptr(dm),
        ptr(workspace), cfg);
}
void Meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, const at::Tensor&) {}

void Probe(const at::Tensor& a, const at::Tensor& dr, at::Tensor out, at::Tensor workspace) {
    auto pTiling = MakeTiling(32, 128, 64, true, false);
    ResidualConfig cfg{};
    cfg.p = pTiling;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto ptr = [](const at::Tensor& x) { return reinterpret_cast<uint8_t*>(x.data_ptr()); };
    launch_kda_matmul_probe(stream, ptr(a), ptr(dr), ptr(out), ptr(workspace), cfg);
}

void Intra(const at::Tensor& q,const at::Tensor& k,const at::Tensor& g,const at::Tensor& b,
    const at::Tensor& l,const at::Tensor& d,const at::Tensor& dq,const at::Tensor& dk,
    const at::Tensor& dg,const at::Tensor& de,at::Tensor oq,at::Tensor ok,at::Tensor og,
    at::Tensor db,at::Tensor flags,at::Tensor workspace,bool joint){
    for(const auto* x:{&q,&k,&b}) TORCH_CHECK(x->is_privateuseone()&&x->is_contiguous()&&x->scalar_type()==at::kBFloat16,"intra BF16 input mismatch");
    for(const at::Tensor* x:std::initializer_list<const at::Tensor*>{&g,&l,&d,&dq,&dk,&dg,&oq,&ok,&og,&db})
        TORCH_CHECK(x->is_privateuseone()&&x->is_contiguous()&&x->scalar_type()==at::kFloat,"intra FP32 input/output mismatch");
    TORCH_CHECK(k.dim()==4&&k.size(1)%64==0&&k.size(3)==128&&q.sizes()==k.sizes()&&g.sizes()==k.sizes(),"intra unsupported shape");
    TORCH_CHECK(!joint||(de.is_privateuseone()&&de.is_contiguous()&&de.scalar_type()==at::kFloat&&de.sizes()==k.sizes()),"intra DE mismatch");
    TORCH_CHECK(flags.scalar_type()==at::kInt&&flags.is_contiguous()&&flags.is_privateuseone(),"intra flags mismatch");
    TORCH_CHECK(workspace.is_privateuseone()&&workspace.scalar_type()==at::kByte&&workspace.numel()>=16*1024*1024,"intra system workspace insufficient");
    static const auto ft=MakeTiling(KDA_INTRA_M,KDA_INTRA_N,64,false,false);
    static const auto bt=MakeTiling(KDA_INTRA_M,KDA_INTRA_N,64,true,false);
    constexpr int64_t tasksPerChunk=(64/KDA_INTRA_M)*(128/KDA_INTRA_N);
    IntraConfig cfg{ft,bt,uint32_t(k.size(0)),uint32_t(k.size(1)),uint32_t(k.size(2)),uint32_t(k.size(0)*k.size(1)/64*k.size(2)*tasksPerChunk),uint32_t(joint)};
    TORCH_CHECK(flags.numel()>=cfg.tasks*8,"intra flag slots insufficient");
    TORCH_CHECK(db.numel()==(128/KDA_INTRA_N)*k.size(0)*k.size(1)*k.size(2),"intra beta partial shape mismatch");
    auto ptr=[](const at::Tensor& x){return reinterpret_cast<uint8_t*>(x.data_ptr());};
    auto stream=c10_npu::getCurrentNPUStream().stream(true);
    launch_kda_intra_native(std::min<uint32_t>(24,(cfg.tasks+1)/2),stream,ptr(q),ptr(k),ptr(g),ptr(b),ptr(l),ptr(d),ptr(dq),ptr(dk),ptr(dg),ptr(de),ptr(oq),ptr(ok),ptr(og),ptr(db),ptr(flags),ptr(workspace),cfg);
}
std::vector<int64_t> IntraConfigInfo(){
    constexpr int64_t m=KDA_INTRA_M,n=KDA_INTRA_N,c=64;
    return {m,n,6*m*c+28*c*n+16*m*n+4096,KDA_INTRA_GLOBAL_ANCHOR};
}
}

TORCH_LIBRARY(kda_residual_native, m) {
    m.def("producer(Tensor value, Tensor residual, Tensor inverse, Tensor dr, Tensor state, "
          "Tensor key, Tensor gate, Tensor beta, Tensor(a!) dk, Tensor(b!) dg, "
          "Tensor(c!) dv, Tensor(d!) db, Tensor(e!) dm, Tensor(f!) workspace) -> ()");
    m.def("probe(Tensor a, Tensor dr, Tensor(a!) out, Tensor(b!) workspace) -> ()");
    m.def("intra(Tensor q, Tensor k, Tensor g, Tensor beta, Tensor l, Tensor d, Tensor dq, Tensor dk, Tensor dg, Tensor de, Tensor(a!) oq, Tensor(b!) ok, Tensor(c!) og, Tensor(d!) db, Tensor(e!) flags, Tensor(f!) workspace, bool joint) -> ()");
    m.def("intra_config() -> int[]",&IntraConfigInfo);
}
TORCH_LIBRARY_IMPL(kda_residual_native, PrivateUse1, m) { m.impl("producer", &Producer); m.impl("probe", &Probe); m.impl("intra", &Intra); }
TORCH_LIBRARY_IMPL(kda_residual_native, Meta, m) { m.impl("producer", &Meta); }
