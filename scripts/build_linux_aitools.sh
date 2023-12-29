#!/bin/bash

# SPDX-FileCopyrightText: 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

SAMPLES_TAG=$1

git clone --depth 1 --branch "$SAMPLES_TAG" https://github.com/Jefino9488/ByteBanter.git

source /intel/oneapi/intelpython/bin/activate tensorflow
cd ByteBanter
python -m pip install -r requirements.txt
python helper.py