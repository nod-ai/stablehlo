#!/usr/bin/env bash

# Needs to run inside the XLA source dir.

set -e
set -o pipefail

XLA_DIR=$1
cd "$XLA_DIR"

# configure
export GCC_HOST_COMPILER_PATH=$(which gcc-10)
export CC=$GCC_HOST_COMPILER_PATH
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_DOWNLOAD_CLANG=0
export CC_OPT_FLAGS="-Wno-sign-compare"
python configure.py

# build
bazel build --verbose_failures --config=monolithic //xla:xla_cc_shared
