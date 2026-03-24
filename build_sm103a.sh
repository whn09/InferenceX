#!/bin/bash
set -euo pipefail

# Build vLLM with native sm_103a support for B300
# Strategy: start from cu130-nightly, rebuild vLLM C extensions with sm_103a,
# then create a new Docker image with the rebuilt wheel.

IMAGE_BASE="vllm/vllm-openai:cu130-nightly"
IMAGE_OUT="vllm-b300-sm103a:latest"
BUILD_DIR="/opt/dlami/nvme/vllm_build"
WHEEL_DIR="/opt/dlami/nvme/vllm_wheelhouse"

echo "============================================"
echo " Building vLLM with sm_103a (B300 native)"
echo "============================================"
echo "Base image: $IMAGE_BASE"
echo "Output:     $IMAGE_OUT"
echo ""

# Step 1: Get vLLM version info
echo ">>> Step 1: Getting vLLM version from base image..."
VLLM_VERSION=$(docker run --rm --entrypoint python3 "$IMAGE_BASE" \
    -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "    vLLM version: $VLLM_VERSION"

# Step 2: Clone vLLM source
echo ">>> Step 2: Cloning vLLM source..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$WHEEL_DIR"
cd "$BUILD_DIR"

git clone https://github.com/vllm-project/vllm.git
cd vllm

# Try to match the nightly commit
if echo "$VLLM_VERSION" | grep -qP '\+g[a-f0-9]+'; then
    SHORT_HASH=$(echo "$VLLM_VERSION" | grep -oP '\+g\K[a-f0-9]+')
    echo "    Checking out commit: $SHORT_HASH"
    git checkout "$SHORT_HASH" 2>/dev/null || echo "    Commit not found, staying on main HEAD"
fi

# Ensure setuptools-scm can detect the version (needs tags)
git fetch --tags 2>/dev/null || true

echo "    Building from commit: $(git rev-parse --short HEAD)"

# Step 3: Build wheel inside container with GPU access (needed for arch detection)
echo ""
echo ">>> Step 3: Building vLLM wheel with TORCH_CUDA_ARCH_LIST=10.0a;10.3a"
echo "    This will take 30-60 minutes..."
echo ""

docker run --rm \
    --gpus all \
    --ipc=host \
    -v "$BUILD_DIR/vllm":/build/vllm:rw \
    -v "$WHEEL_DIR":/wheelhouse:rw \
    -e "TORCH_CUDA_ARCH_LIST=10.0a;10.3a" \
    -e VLLM_TARGET_DEVICE=cuda \
    -e MAX_JOBS=48 \
    -e SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" \
    -w /build/vllm \
    --entrypoint bash \
    "$IMAGE_BASE" \
    -c '
set -ex

echo "=== Build Environment ==="
python3 -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.version.cuda)"
nvcc --version | tail -1
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
nvidia-smi -L | head -1

# Install system deps (git needed for CMake FetchContent/cutlass)
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1

# Fix missing CUDA dev headers - symlink from pip-installed packages
for pkg_dir in /usr/local/lib/python3.12/dist-packages/nvidia/*/include; do
    if [ -d "$pkg_dir" ]; then
        ln -sf "$pkg_dir"/*.h /usr/local/cuda/include/ 2>/dev/null || true
        # Also copy subdirs (e.g. cusparse/)
        for subdir in "$pkg_dir"/*/; do
            if [ -d "$subdir" ]; then
                cp -rsn "$subdir" /usr/local/cuda/include/ 2>/dev/null || true
            fi
        done
    fi
done
echo "CUDA headers available: $(ls /usr/local/cuda/include/cublas_v2.h /usr/local/cuda/include/cusparse.h 2>/dev/null | wc -l) key headers"

# Fix missing nvrtc symlink for CMake
ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so 2>/dev/null || true

# Patch CMakeLists.txt to add 10.3 to supported archs whitelist
sed -i "s/10.0;11.0;12.0/10.0;10.3;11.0;12.0/g" CMakeLists.txt
echo "Patched CMakeLists.txt:"
grep CUDA_SUPPORTED_ARCHS CMakeLists.txt | head -2

# Install build deps
pip install --no-cache-dir build "cmake>=3.26" ninja packaging "setuptools-scm>=8" wheel jinja2

# Build the wheel
echo ""
echo "=== Building vLLM wheel ==="
python3 setup.py bdist_wheel --dist-dir /wheelhouse 2>&1

echo ""
echo "=== Wheel built ==="
ls -lh /wheelhouse/*.whl
'

echo ""
echo ">>> Wheel built successfully:"
ls -lh "$WHEEL_DIR"/*.whl

# Step 4: Create new Docker image
echo ""
echo ">>> Step 4: Creating Docker image with sm_103a wheel..."

# Copy wheels into build context
cp -r "$WHEEL_DIR" "$BUILD_DIR/wheelhouse"

cat > "$BUILD_DIR/Dockerfile" << 'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install the rebuilt vLLM wheel with sm_103a native kernels
COPY wheelhouse/*.whl /tmp/
RUN pip uninstall -y vllm 2>/dev/null; \
    pip install --no-cache-dir --no-deps /tmp/vllm*.whl && \
    rm -f /tmp/vllm*.whl

ENV TORCH_CUDA_ARCH_LIST="10.0a;10.3a"
DOCKERFILE

docker build \
    --build-arg BASE_IMAGE="$IMAGE_BASE" \
    -f "$BUILD_DIR/Dockerfile" \
    -t "$IMAGE_OUT" \
    "$BUILD_DIR"

echo ""
echo "============================================"
echo " Build Complete!"
echo "============================================"
docker images "$IMAGE_OUT"

# Step 5: Verify sm_103a kernels
echo ""
echo ">>> Step 5: Verifying sm_103a native kernels..."
docker run --rm --gpus all --entrypoint bash "$IMAGE_OUT" -c '
python3 -c "import vllm; print(\"vLLM version:\", vllm.__version__)"

echo ""
echo "=== Checking CUDA archs in vLLM _C.so ==="
SO=$(find /usr/local/lib/python3.12/dist-packages/vllm/ -name "_C*.so" 2>/dev/null | head -1)
if [ -n "$SO" ]; then
    SM100=$(cuobjdump --list-elf "$SO" 2>/dev/null | grep -c "sm_100" || echo 0)
    SM103=$(cuobjdump --list-elf "$SO" 2>/dev/null | grep -c "sm_103" || echo 0)
    echo "  sm_100 kernels: $SM100"
    echo "  sm_103 kernels: $SM103"
    if [ "$SM103" -gt 0 ]; then
        echo "  SUCCESS: Native sm_103a kernels found!"
    else
        echo "  WARNING: No sm_103a kernels found"
    fi
else
    echo "  ERROR: _C.so not found"
fi
'

echo ""
echo "============================================"
echo " Image ready: $IMAGE_OUT"
echo " Run: docker run --gpus all $IMAGE_OUT vllm serve ..."
echo "============================================"
