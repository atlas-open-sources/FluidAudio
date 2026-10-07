#!/bin/zsh
setopt pipefail
cd ~/src/nemotron-export
rm -f P4B_DONE
{
  .venv/bin/python palettize_encoder.py --model-dir out/320ms-fp16 --output-dir out/320ms-4bit-pcs --per-channel-scale 2>&1 | tail -2
  ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/320ms-4bit-pcs --languages en_us --samples 50 --output benchres/bench_320ms-4bit-pcs.json 2>&1 | grep -E "CER=" | tail -1
} 
{
  .venv/bin/python palettize_encoder.py --model-dir out/320ms-fp16 --output-dir out/320ms-6bit --nbits 6 2>&1 | tail -2
  ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/320ms-6bit --languages en_us --samples 50 --output benchres/bench_320ms-6bit.json 2>&1 | grep -E "CER=" | tail -1
}
touch P4B_DONE
