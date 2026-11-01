ARG BASE_IMAGE=zhangyi2026/residual-wy-h100@sha256:5f6e15aa35959132c609d5d4f823b798f0821b8b961bf91d9ac766a75e41c0ac
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.version="20261031-cu126-triton371-statefix1" \
      org.opencontainers.image.description="Fix D256 optimized DeltaNet state shared-memory overflow with per-launch synchronous MMA lowering"
COPY patch_payload.py check_state_codegen.py /opt/hopper-statefix/
RUN python /opt/hopper-statefix/patch_payload.py && \
    python bench.py verify && \
    H100_IMAGE_BUILD_TEST=1 H100_IMAGE_IMPORT_TEST=1 python -m unittest test_image -v && \
    python -m unittest discover -s single -p test_local.py -v
