#!/bin/zsh
setopt pipefail
set -e
cd ~/src/nemotron-export
rm -f P4_DONE P4_FAILED
{
  .venv/bin/python palettize_encoder.py --model-dir out/320ms-fp16 --output-dir out/320ms-4bit --granularity per_grouped_channel --group-size 16 2>&1 | tail -4
  ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/320ms-4bit --languages en_us,cmn_hans_cn --samples 100 --output benchres/bench_320ms-4bit.json 2>&1 | grep -E "CER=" | tail -2
  touch P4_DONE
} || touch P4_FAILED
