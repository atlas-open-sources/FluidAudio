"""Probe: can we load the prompted multilingual checkpoint, and can we set lookahead?"""
import torch, nemo.collections.asr as nemo_asr
from nemo.core import ModelPT

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

print("== loading (auto-detect class) ==")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID, map_location="cpu")
model.eval()
print("class:", type(model).__module__ + "." + type(model).__name__)

enc = model.encoder
print("encoder class:", type(enc).__name__)
for attr in ["att_context_size", "att_context_style", "streaming_cfg", "subsampling_factor"]:
    print(f"  {attr}:", getattr(enc, attr, "<none>"))

print("== can we set the lookahead? ==")
print("  has set_default_att_context_size:", hasattr(enc, "set_default_att_context_size"))
print("  has setup_streaming_params:", hasattr(enc, "setup_streaming_params"))
import inspect
if hasattr(enc, "setup_streaming_params"):
    print("  setup_streaming_params signature:", inspect.signature(enc.setup_streaming_params))
print("  cfg att_context_size:", model.cfg.encoder.get("att_context_size"))
print("  cfg att_context_probs:", model.cfg.encoder.get("att_context_probs"))
