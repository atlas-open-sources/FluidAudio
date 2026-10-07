#!/bin/zsh
setopt pipefail
set -e
cd ~/src/nemotron-export
REF="$HOME/Library/Application Support/FluidAudio/Models/nemotron-multilingual/latin/560ms/tokenizer.json"
rm -f L560_DONE L560_FAILED
{
  .venv/bin/python convert_nemotron_streaming.py --output-dir out/latin-560ms-fp16 --lookahead 6 --precision FLOAT16 --prune-tokenizer "$REF" 2>&1 | grep -E "lookahead=|pruned|Done!" | tail -3
  rm -rf out/latin-560ms-ane && mkdir -p out/latin-560ms-ane
  cp -R out/560ms-ane/encoder.mlpackage out/latin-560ms-ane/
  for f in decoder.mlpackage joint.mlpackage preprocessor.mlpackage metadata.json tokenizer.json; do
    cp -R out/latin-560ms-fp16/$f out/latin-560ms-ane/$f
  done
  cp -R out/latin-320ms-ane/decoder_joint.mlpackage out/latin-560ms-ane/
  for name in latin-560ms-ane latin-320ms-ane; do
    ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/$name --languages en_us --samples 100 --output benchres/bench_${name}-fused.json 2>&1 | grep -E "CER=" | tail -1
  done
  touch L560_DONE
} || touch L560_FAILED
