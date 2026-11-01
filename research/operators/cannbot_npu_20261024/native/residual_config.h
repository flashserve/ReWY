#pragma once
#include <cstdint>
#include "kernel_tiling.h"
#ifndef KDA_INTRA_M
#define KDA_INTRA_M 16
#endif
#ifndef KDA_INTRA_N
#define KDA_INTRA_N 32
#endif
#ifndef KDA_INTRA_GLOBAL_ANCHOR
#define KDA_INTRA_GLOBAL_ANCHOR 0
#endif
static_assert((KDA_INTRA_M==16||KDA_INTRA_M==32)&&(KDA_INTRA_N==32||KDA_INTRA_N==64),"unsupported intra tile");

struct ResidualConfig {
    AscendC::tiling::TCubeTiling p;
    AscendC::tiling::TCubeTiling consume;
    uint32_t batch;
    uint32_t time;
    uint32_t heads;
    uint32_t keyHeads;
    uint32_t tasks;
};

struct IntraConfig {
    AscendC::tiling::TCubeTiling forward;
    AscendC::tiling::TCubeTiling backward;
    uint32_t batch, time, heads, tasks, joint;
};
