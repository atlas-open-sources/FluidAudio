#!/bin/zsh
setopt pipefail
set -e
cd ~/src/nemotron-export
rm -f FUSED_DONE FUSED_FAILED
{
  .venv/bin/python export_decoder_joint.py --dest out/320ms-ane --dest out/560ms-ane --dest out/80ms-ane --dest out/320ms-fp16 --dest out/560ms-fp16 --dest out/80ms-fp16 2>&1 | tail -8
  .venv/bin/python export_decoder_joint.py --dest out/latin-320ms-ane --prune-tokenizer "$HOME/Library/Application Support/FluidAudio/Models/nemotron-multilingual/latin/560ms/tokenizer.json" 2>&1 | tail -4
  for name in 560ms-ane 320ms-ane; do
    ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/$name --languages en_us,cmn_hans_cn --samples 100 --output benchres/bench_${name}-fused.json 2>&1 | grep -E "CER=" | tail -2
  done
  touch FUSED_DONE
} || touch FUSED_FAILED
