#!/bin/bash
# Build vLLM Docker image with EFA + GDRCopy + Mooncake (USE_EFA=ON)
# Run on each node (prefill and decode).
#
# Prerequisites:
#   - Base image: vllm-b300-sm103a:latest (vLLM v0.18.0 rebuilt with sm_103a)
#   - Or: vllm/vllm-openai:v0.18.0-cu130 (will use PTX JIT, slower)
#
# Usage: bash build_efa_image.sh [--base-image IMAGE]

BASE_IMAGE="${BASE_IMAGE:-vllm-b300-sm103a:latest}"
OUTPUT_TAG="${OUTPUT_TAG:-vllm-b300-sm103a:latest}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        --output-tag) OUTPUT_TAG="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Building EFA image from $BASE_IMAGE -> $OUTPUT_TAG"

docker build -t "$OUTPUT_TAG" -f - . << 'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl pkg-config \
    libgflags-dev libgoogle-glog-dev libjsoncpp-dev \
    libnuma-dev libibverbs-dev libboost-all-dev \
    libcurl4-openssl-dev libyaml-cpp-dev libgtest-dev \
    pybind11-dev environment-modules tcl \
    pciutils libzstd-dev libxxhash-dev libmsgpack-dev \
    && rm -rf /var/lib/apt/lists/*

# GDRCopy 2.5.1
RUN cd /tmp && wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.1.tar.gz && \
    tar xf v2.5.1.tar.gz && cd gdrcopy-2.5.1 && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy*

# AWS EFA 1.47.0 with GDR
RUN cd /tmp && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz && \
    tar xzf aws-efa-installer-1.47.0.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod -g --no-verify && \
    rm -rf /tmp/aws-efa-installer*

# yalantinglibs (Mooncake dependency)
RUN cd /tmp && git clone --depth 1 https://github.com/alibaba/yalantinglibs.git && \
    cd yalantinglibs && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc) && make install && \
    rm -rf /tmp/yalantinglibs

# Mooncake Transfer Engine with EFA support
RUN cd /tmp && git clone https://github.com/kvcache-ai/Mooncake.git && cd Mooncake && \
    git submodule update --init --recursive && mkdir build && cd build && \
    export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH && \
    cmake .. -DUSE_EFA=ON -DUSE_CUDA=ON -DWITH_TE=ON -DWITH_STORE=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DPython3_EXECUTABLE=$(which python3) \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DPYBIND11_PYTHON_VERSION=3.12 && \
    make -j$(nproc) && \
    cp mooncake-integration/engine.cpython-*.so ../mooncake-wheel/mooncake/ && \
    cp mooncake-asio/libasio.so ../mooncake-wheel/mooncake/ && \
    pip install ../mooncake-wheel --no-build-isolation --force-reinstall && \
    rm -rf /tmp/Mooncake

# Verify Mooncake installation
RUN python3 -c "from mooncake.engine import TransferEngine; print('TransferEngine OK')"

# EFA environment
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/amazon/efa/bin:${PATH}"
DOCKERFILE
