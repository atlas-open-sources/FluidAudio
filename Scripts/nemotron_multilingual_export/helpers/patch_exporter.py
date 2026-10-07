"""Parameterise FluidInference's Nemotron CoreML exporter by lookahead.

Two things the published script hardcodes stop it working for us:
  * it loads EncDecRNNTBPEModel, but the multilingual checkpoint is
    EncDecRNNTBPEModelWithPrompt (a 128-entry language-prompt table);
  * it bakes in the 1120 ms shapes (112 mel frames, 14 encoder frames), so any
    other chunk size would export cleanly and be wrong.
"""
import pathlib

p = pathlib.Path("convert_nemotron_streaming.py")
s = p.read_text()
orig = s

s = s.replace(
    'DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"',
    'DEFAULT_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"',
)

s = s.replace(
    'model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")',
    '# ASRModel auto-detects the concrete class. This checkpoint is\n'
    '    # EncDecRNNTBPEModelWithPrompt, NOT the unprompted class the English\n'
    '    # export hardcoded.\n'
    '    model = nemo_asr.models.ASRModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")',
)

s = s.replace(
    '    precision: str = typer.Option("FLOAT32", help="FLOAT32 or FLOAT16"),',
    '    precision: str = typer.Option("FLOAT32", help="FLOAT32 or FLOAT16"),\n'
    '    lookahead: int = typer.Option(6, help="Encoder lookahead tokens: 0=80ms, 3=320ms, 6=560ms, 13=1120ms"),',
)

DERIVE = '''supported = [c[1] for c in model.cfg.encoder.att_context_size]
    if lookahead not in supported:
        raise typer.BadParameter(f"lookahead {lookahead} not in checkpoint supported {supported}")
    left = int(model.cfg.encoder.att_context_size[0][0])
    encoder.set_default_att_context_size([left, lookahead])
    encoder.setup_streaming_params(att_context_size=[left, lookahead])

    # DERIVED, not hardcoded. The published script baked in the 1120 ms numbers;
    # reading them back off streaming_cfg is what makes another lookahead
    # correct rather than merely exportable.
    scfg = encoder.streaming_cfg

    def _last(v):
        return int(v[-1]) if isinstance(v, (list, tuple)) else int(v)

    global CHUNK_MEL_FRAMES, PRE_ENCODE_CACHE, TOTAL_MEL_FRAMES
    CHUNK_MEL_FRAMES = _last(scfg.chunk_size)
    PRE_ENCODE_CACHE = _last(scfg.pre_encode_cache_size)
    TOTAL_MEL_FRAMES = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE
    chunk_frames = int(scfg.valid_out_len)
    typer.echo(
        f"lookahead={lookahead} att_context=[{left},{lookahead}] "
        f"chunk_mel={CHUNK_MEL_FRAMES} pre_encode_cache={PRE_ENCODE_CACHE} "
        f"total_mel={TOTAL_MEL_FRAMES} enc_frames={chunk_frames} "
        f"chunk_ms={CHUNK_MEL_FRAMES * 10}"
    )'''
s = s.replace('    encoder.setup_streaming_params()', '    ' + DERIVE)

s = s.replace('        chunk_size_frames=14,', '        chunk_size_frames=chunk_frames,')

s = s.replace(
    '        "encoder_dim": int(enc_out.shape[1]),',
    '        "encoder_dim": int(enc_out.shape[1]),\n'
    '        "att_context_size": [int(left), int(lookahead)],\n'
    '        "chunk_ms": int(CHUNK_MEL_FRAMES * 10),\n'
    '        "model_class": type(model).__module__ + "." + type(model).__name__,',
)

assert s != orig, "no replacements applied"
for marker in ["ASRModel.from_pretrained", "lookahead: int", "scfg.valid_out_len",
               "chunk_size_frames=chunk_frames", '"chunk_ms"']:
    assert marker in s, f"missing after patch: {marker}"
p.write_text(s)
print("patched OK")
