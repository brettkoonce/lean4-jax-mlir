#!/bin/bash
export HIP_VISIBLE_DEVICES=0
export IREE_BACKEND=rocm
exec .lake/build/bin/efficientnet-v2-train 2>&1 | tee effnet_v2.log
