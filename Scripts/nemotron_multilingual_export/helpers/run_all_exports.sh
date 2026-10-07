#!/bin/zsh
set -e
cd ~/src/nemotron-export
.venv/bin/python convert_nemotron_streaming.py --output-dir out/560ms --lookahead 6 2>&1 | tee export_560_prompt.log
.venv/bin/python convert_nemotron_streaming.py --output-dir out/320ms --lookahead 3 2>&1 | tee export_320_prompt.log
.venv/bin/python convert_nemotron_streaming.py --output-dir out/80ms --lookahead 0 2>&1 | tee export_80_prompt.log
touch EXPORTS_DONE
