#!/bin/zsh
# Tier x precision matrix for Nemotron multilingual CoreML exports.
# Phase 1: FP16 exports + int8 quantization. Phase 2: FLEURS en+zh benchmark.
set -e
cd ~/src/nemotron-export
rm -f MATRIX_DONE MATRIX_FAILED

phase1() {
  local t la pair
  for pair in "560 6" "320 3" "80 0"; do
    t=${pair%% *}; la=${pair##* }
    if [ ! -d out/${t}ms-fp16/encoder.mlpackage ]; then
      echo "### export FP16 ${t}ms"
      .venv/bin/python convert_nemotron_streaming.py \
        --output-dir out/${t}ms-fp16 --lookahead $la --precision FLOAT16 \
        2>&1 | tee export_${t}_fp16.log
    fi
    if [ ! -d out/${t}ms-int8/encoder.mlpackage ]; then
      echo "### quantize int8 ${t}ms"
      .venv/bin/python quantize_encoder.py \
        --model-dir out/${t}ms --output-dir out/${t}ms-int8 \
        2>&1 | tee quant_${t}.log
    fi
    for f in decoder.mlpackage joint.mlpackage preprocessor.mlpackage metadata.json tokenizer.json; do
      [ -e out/${t}ms-int8/$f ] || cp -R out/${t}ms/$f out/${t}ms-int8/$f
    done
  done
}

phase2() {
  local CLI=~/src/FluidAudio/.build/debug/fluidaudiocli
  mkdir -p benchres
  local dirs=(560ms 320ms 80ms 560ms-fp16 320ms-fp16 80ms-fp16 560ms-int8 320ms-int8 80ms-int8)
  for name in $dirs; do
    if [ ! -f benchres/bench_${name}.json ]; then
      echo "### bench $name"
      $CLI nemotron-multilingual-benchmark --model-dir out/$name \
        --languages en_us,cmn_hans_cn --samples 100 \
        --output benchres/bench_${name}.json 2>&1 | tee benchres/bench_${name}.log \
        || echo "BENCH_ROW_FAILED $name"
    fi
  done
  # Shipped baselines: latin/560 (en only - vocab-pruned) + multilingual/1120 (en+zh)
  local ship_latin="$HOME/Library/Application Support/FluidAudio/Models/nemotron-multilingual/latin/560ms"
  if [ -d "$ship_latin" ] && [ ! -f benchres/bench_shipped_latin560.json ]; then
    echo "### bench shipped latin/560"
    $CLI nemotron-multilingual-benchmark --model-dir "$ship_latin" \
      --languages en_us --samples 100 \
      --output benchres/bench_shipped_latin560.json 2>&1 | tee benchres/bench_shipped_latin560.log \
      || echo "BENCH_ROW_FAILED shipped_latin560"
  fi
  # pull the shipped multilingual/1120 via one transcribe call (auto-download), then bench it
  local ship_ml="$HOME/Library/Application Support/FluidAudio/Models/nemotron-multilingual/multilingual/1120ms"
  if [ ! -d "$ship_ml" ]; then
    $CLI nemotron-multilingual-transcribe --language zh-CN --chunk-ms 1120 \
      --input ami-es2004a-multispeaker-16k.wav >/dev/null 2>&1 || true
  fi
  if [ -d "$ship_ml" ] && [ ! -f benchres/bench_shipped_ml1120.json ]; then
    echo "### bench shipped multilingual/1120"
    $CLI nemotron-multilingual-benchmark --model-dir "$ship_ml" \
      --languages en_us,cmn_hans_cn --samples 100 \
      --output benchres/bench_shipped_ml1120.json 2>&1 | tee benchres/bench_shipped_ml1120.log \
      || echo "BENCH_ROW_FAILED shipped_ml1120"
  fi
}

if phase1 && phase2; then
  touch MATRIX_DONE
else
  touch MATRIX_FAILED
fi
