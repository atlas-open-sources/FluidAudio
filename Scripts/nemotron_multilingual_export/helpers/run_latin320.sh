#!/bin/zsh
setopt pipefail
set -e
cd ~/src/nemotron-export
REF="$HOME/Library/Application Support/FluidAudio/Models/nemotron-multilingual/latin/560ms/tokenizer.json"
rm -f LATIN_DONE LATIN_FAILED
{
  .venv/bin/python convert_nemotron_streaming.py --output-dir out/latin-320ms-fp16 --lookahead 3 --precision FLOAT16 --prune-tokenizer "$REF" 2>&1 | tee export_latin320.log
  .venv/bin/python quantize_encoder.py --model-dir out/latin-320ms-fp16 --output-dir out/latin-320ms-ane 2>&1 | tail -5
  for f in decoder.mlpackage joint.mlpackage preprocessor.mlpackage metadata.json tokenizer.json; do
    [ -e out/latin-320ms-ane/$f ] || cp -R out/latin-320ms-fp16/$f out/latin-320ms-ane/$f
  done
  ~/src/FluidAudio/.build/debug/fluidaudiocli nemotron-multilingual-benchmark --model-dir out/latin-320ms-ane --languages en_us --samples 100 --output benchres/bench_latin-320ms-ane.json 2>&1 | tail -3
  touch LATIN_DONE
} || touch LATIN_FAILED
