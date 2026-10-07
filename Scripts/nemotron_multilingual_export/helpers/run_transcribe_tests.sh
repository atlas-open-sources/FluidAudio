#!/bin/zsh
cd ~/src/nemotron-export
CLI=~/src/FluidAudio/.build/release/fluidaudiocli
WAV=~/src/nemotron-export/ami-es2004a-multispeaker-16k.wav
rm -f TESTS_DONE
echo "=== baseline shipped 560 (auto-download) ==="
$CLI nemotron-multilingual-transcribe --language en-US --chunk-ms 560 --input $WAV 2>&1 | tail -25 | tee t_shipped560.log
for tier in 560 320 80; do
  echo "=== ours ${tier}ms ==="
  $CLI nemotron-multilingual-transcribe --model-dir out/${tier}ms --language en-US --input $WAV 2>&1 | tail -25 | tee t_ours${tier}.log
done
touch TESTS_DONE
