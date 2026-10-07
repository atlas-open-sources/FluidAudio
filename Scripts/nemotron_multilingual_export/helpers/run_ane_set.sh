#!/bin/zsh
setopt pipefail
set -e
cd ~/src/nemotron-export
rm -f ANE_SET_DONE ANE_SET_FAILED
{
  for t in 560 80; do
    if [ ! -d out/${t}ms-ane/encoder.mlpackage ]; then
      .venv/bin/python quantize_encoder.py --model-dir out/${t}ms-fp16 --output-dir out/${t}ms-ane 2>&1 | tail -3
      for f in decoder.mlpackage joint.mlpackage preprocessor.mlpackage metadata.json tokenizer.json; do
        [ -e out/${t}ms-ane/$f ] || cp -R out/${t}ms-fp16/$f out/${t}ms-ane/$f
      done
    fi
    ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/${t}ms-ane --languages en_us,cmn_hans_cn --samples 100 --output benchres/bench_${t}ms-ane.json 2>&1 | tail -2
  done
  touch ANE_SET_DONE
} || touch ANE_SET_FAILED
