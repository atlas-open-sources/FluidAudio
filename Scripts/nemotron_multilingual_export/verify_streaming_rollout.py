#!/usr/bin/env python3
"""Immutable-package, real-recording streaming qualification (not WER acceptance).

Replicates the Swift fixed-shape chunk convention, including zero pre-encode
padding on the first chunk and zero audio tail padding. It does not claim to
replicate NeMo's variable-shaped first-chunk streaming API.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from qualification_provenance import measure_difference, runtime_provenance, sha256


def prepend_history(mel, previous, frames, history_frames):
    if mel.ndim != 3 or mel.shape[0] != 1 or frames < 1 or history_frames < 1:
        raise ValueError('Invalid mel shape or frame configuration')
    if mel.shape[2] < history_frames:
        raise ValueError('Insufficient mel frames for history')
    result = np.zeros((1, mel.shape[1], frames + history_frames), dtype=np.float32)
    if previous is not None:
        if previous.shape != (1, mel.shape[1], history_frames):
            raise ValueError('History shape mismatch')
        result[:, :, :history_frames] = previous
    take = min(frames, mel.shape[2])
    result[:, :, history_frames:history_frames + take] = mel[:, :, :take]
    # Match Swift extractMelCachePure: last frames from the whole preprocessor result.
    return result, mel[:, :, -history_frames:].copy()


def validated_frame_count(encoded, length):
    count = int(np.asarray(length).reshape(-1)[0])
    if encoded.ndim != 3 or count < 1 or count > encoded.shape[2]:
        raise ValueError('Invalid encoded length')
    return count


def next_prediction_state(token, candidate_h, candidate_c, previous, blank):
    # RNNT blanks advance time but must not commit candidate recurrent state.
    return previous if token == blank else (token, candidate_h, candidate_c)


def verify_manifest(directory, receipt):
    hashes = receipt.get('output_sha256', {})
    if not hashes:
        raise ValueError('Missing immutable model hash manifest')
    for name, expected in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe manifest path')
        if sha256(directory / name) != expected:
            raise ValueError(f'Immutable export changed: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audio', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='auto')
    parser.add_argument('--max-symbols', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    if not 1 <= args.max_symbols <= 100:
        raise ValueError('Invalid max-symbols')
    receipt_path = args.model_dir / 'export-receipt.json'
    parent = json.loads(receipt_path.read_text())
    verify_manifest(args.model_dir, parent)
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != parent['checkpoint']['sha256']:
        raise ValueError('Checkpoint mismatch')
    meta = json.loads((args.model_dir / 'metadata.json').read_text())
    if args.prompt not in meta['prompt_dictionary']:
        raise ValueError('Unsupported prompt')
    if meta.get('vocab_pruned'):
        raise ValueError('This verifier requires the full checkpoint vocabulary')
    if len(set(p.resolve() for p in args.audio)) != len(args.audio):
        raise ValueError('Duplicate recordings')
    args.output.mkdir(parents=True)
    report = dict(schema_version=1, status='started', **runtime_provenance(),
                  model_directory=str(args.model_dir.resolve()),
                  export_receipt_sha256=sha256(receipt_path),
                  source_export_status=parent['status'], checkpoint_sha256=checkpoint_sha,
                  compute_units={k: 'CPU_ONLY' for k in ['preprocessor', 'encoder', 'decoder', 'joint']},
                  torch_threads=1, seed=42, max_symbols_per_frame=args.max_symbols,
                  prompt=args.prompt, prompt_id=meta['prompt_dictionary'][args.prompt],
                  convention='Swift fixed-shape chunks, zero initial pre-encode history and audio tail padding; not NeMo variable-first-chunk API',
                  gate='Every float normalized RMSE <=0.05, exact integer lengths, teacher joint top1 and independent greedy token equality; no WER claim',
                  recordings=[])
    report_path = args.output / 'rollout-receipt.json'
    report_path.write_text(json.dumps(report, indent=2))
    try:
        run(args, meta, report)
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


def run(args, meta, report):
    import coremltools as ct
    import torch
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from individual_components import PreprocessorWrapper, EncoderStreamingWrapper, DecoderWrapper, JointWrapper

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    started = time.monotonic()
    model = nemo_asr.models.ASRModel.restore_from(str(args.checkpoint), map_location='cpu').eval()
    if int(model.tokenizer.vocab_size) != meta['vocab_size'] or not hasattr(model, 'prompt_kernel'):
        raise ValueError('Wrong checkpoint class or vocabulary')
    # Match exported prediction network: no training SOS prepended to each token.
    model.decoder._rnnt_export = True
    model.encoder.set_default_att_context_size(meta['att_context_size'])
    model.encoder.setup_streaming_params(att_context_size=meta['att_context_size'])
    wrappers = {
        'preprocessor': PreprocessorWrapper(model.preprocessor).eval(),
        'encoder': EncoderStreamingWrapper(model.encoder, model.prompt_kernel, meta['num_prompts']).eval(),
        'decoder': DecoderWrapper(model.decoder).eval(),
        'joint': JointWrapper(model.joint).eval(),
    }
    models = {name: ct.models.MLModel(str(args.model_dir / (name + '.mlpackage')), compute_units=ct.ComputeUnit.CPU_ONLY)
              for name in wrappers}
    report['load_seconds'] = time.monotonic() - started
    output_names = {'preprocessor': ['mel', 'mel_length'], 'encoder': ['encoded', 'encoded_length', 'cache_channel_out', 'cache_time_out', 'cache_len_out'],
                    'decoder': ['decoder_out', 'h_out', 'c_out'], 'joint': ['logits']}
    failed = False
    event_count = 0
    maximum = {}
    log = (args.output / 'steps.jsonl').open('w')

    def source(name, inputs):
        with torch.inference_mode():
            values = wrappers[name](*(torch.from_numpy(np.asarray(v).copy()) for v in inputs.values()))
        if isinstance(values, torch.Tensor):
            values = (values,)
        return {key: value.detach().numpy() for key, value in zip(output_names[name], values)}

    def compare(kind, expected, actual, **location):
        nonlocal failed, event_count
        checks = {key: measure_difference(value, np.asarray(actual[key])) for key, value in expected.items()}
        for key, check in checks.items():
            metric = kind + '/' + key
            maximum[metric] = max(maximum.get(metric, 0), check['normalized_rmse'])
        passed = all(row['passed'] for row in checks.values())
        failed |= not passed
        log.write(json.dumps(dict(kind=kind, **location, checks=checks)) + '\n')
        log.flush()
        event_count += 1

    def decoder_inputs(state):
        token, h, c = state
        return {'token': np.array([[token]], np.int32), 'token_length': np.array([1], np.int32), 'h_in': h, 'c_in': c}

    try:
        for audio_path in args.audio:
            audio, rate = sf.read(audio_path, dtype='float32')
            if rate != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError('Require finite nonempty mono16k real WAV')
            chunk_samples = meta['chunk_mel_frames'] * 160
            if len(audio) <= chunk_samples:
                raise ValueError('Require at least two chunks to verify warmed caches')
            row = dict(path=str(audio_path.resolve()), sha256=sha256(audio_path), samples=len(audio), seconds=len(audio) / rate,
                       chunks=0, source_tokens=[], coreml_tokens=[], teacher_top1_mismatches=0,
                       nonblank_teacher_decoder_calls=0, nonzero_teacher_state_calls=0,
                       source_symbol_limit_hits=0, coreml_symbol_limit_hits=0)
            report['recordings'].append(row)
            channels, times, lengths = model.encoder.get_initial_cache_state(batch_size=1, device='cpu')
            cache = {'cache_channel': channels.transpose(0, 1).numpy(), 'cache_time': times.transpose(0, 1).numpy(),
                     'cache_len': lengths.to(torch.int32).numpy()}
            own_cache = {k: v.copy() for k, v in cache.items()}
            history = own_history = None
            shape = (meta['decoder_layers'], 1, meta['decoder_hidden'])
            state = (meta['blank_idx'], np.zeros(shape, np.float32), np.zeros(shape, np.float32))
            own_state = (state[0], state[1].copy(), state[2].copy())
            for chunk, offset in enumerate(range(0, len(audio), chunk_samples)):
                samples = audio[offset:offset + chunk_samples]
                padded = np.pad(samples, (0, chunk_samples - len(samples)))[None, :]
                inp = {'audio': padded, 'audio_length': np.array([chunk_samples], np.int32)}
                pre = source('preprocessor', inp)
                own_pre = models['preprocessor'].predict(inp)
                loc = dict(recording=audio_path.name, chunk=chunk)
                compare('preprocessor', pre, own_pre, **loc)
                mel, history = prepend_history(pre['mel'], history, meta['chunk_mel_frames'], meta['pre_encode_cache'])
                own_mel, own_history = prepend_history(own_pre['mel'], own_history, meta['chunk_mel_frames'], meta['pre_encode_cache'])
                enc_in = {'mel': mel, 'mel_length': np.array([meta['total_mel_frames']], np.int32), **cache,
                          'prompt_id': np.array([report['prompt_id']], np.int32)}
                ref_enc = source('encoder', enc_in)
                teacher_enc = models['encoder'].predict(enc_in)
                own_enc = models['encoder'].predict({**enc_in, 'mel': own_mel, **own_cache})
                compare('encoder_teacher_forced', ref_enc, teacher_enc, **loc)
                compare('encoder_own_cache_and_preprocessor', ref_enc, own_enc, **loc)
                cache = {key: ref_enc[key + '_out'].copy() for key in cache}
                own_cache = {key: own_enc[key + '_out'].copy() for key in own_cache}
                frames = validated_frame_count(ref_enc['encoded'], ref_enc['encoded_length'])
                own_frames = validated_frame_count(own_enc['encoded'], own_enc['encoded_length'])
                for frame in range(frames):
                    for symbol in range(args.max_symbols):
                        dec_in = decoder_inputs(state)
                        dec = source('decoder', dec_in)
                        teacher_dec = models['decoder'].predict(dec_in)
                        step = dict(**loc, frame=frame, symbol=symbol)
                        compare('decoder_teacher_forced', dec, teacher_dec, **step)
                        row['nonblank_teacher_decoder_calls'] += state[0] != meta['blank_idx']
                        row['nonzero_teacher_state_calls'] += bool(np.any(state[1]) or np.any(state[2]))
                        joint_in = {'encoder': ref_enc['encoded'][:, :, frame:frame + 1].copy(),
                                    'decoder': dec['decoder_out'][:, :, :1].copy()}
                        joint = source('joint', joint_in)
                        teacher_joint = models['joint'].predict(joint_in)
                        compare('joint_teacher_forced', joint, teacher_joint, **step)
                        token = int(np.argmax(joint['logits']))
                        teacher_token = int(np.argmax(teacher_joint['logits']))
                        mismatch = token != teacher_token
                        row['teacher_top1_mismatches'] += mismatch
                        failed |= mismatch
                        log.write(json.dumps(dict(kind='joint_top1', **step, source_token=token, coreml_token=teacher_token)) + '\n')
                        state = next_prediction_state(token, dec['h_out'], dec['c_out'], state, meta['blank_idx'])
                        if token == meta['blank_idx']:
                            break
                        row['source_tokens'].append(token)
                    else:
                        row['source_symbol_limit_hits'] += 1
                for frame in range(own_frames):
                    for symbol in range(args.max_symbols):
                        dec = models['decoder'].predict(decoder_inputs(own_state))
                        joint = models['joint'].predict({'encoder': own_enc['encoded'][:, :, frame:frame + 1].copy(),
                                                        'decoder': dec['decoder_out'][:, :, :1].copy()})
                        token = int(np.argmax(joint['logits']))
                        own_state = next_prediction_state(token, dec['h_out'], dec['c_out'], own_state, meta['blank_idx'])
                        if token == meta['blank_idx']:
                            break
                        row['coreml_tokens'].append(token)
                    else:
                        row['coreml_symbol_limit_hits'] += 1
                row['chunks'] += 1
                print(json.dumps(dict(recording=audio_path.name, chunks=row['chunks'], failed_so_far=failed)), flush=True)
            row['tokens_equal'] = row['source_tokens'] == row['coreml_tokens']
            row['state_coverage_passed'] = row['nonblank_teacher_decoder_calls'] > 0 and row['nonzero_teacher_state_calls'] > 0
            failed |= not row['tokens_equal'] or not row['state_coverage_passed'] or bool(row['source_symbol_limit_hits'] + row['coreml_symbol_limit_hits'])
        report.update(status='numerical_or_token_check_failed' if failed else 'passed',
                      comparison_events=event_count, maximum_normalized_rmse=maximum,
                      elapsed_seconds=time.monotonic() - started)
    finally:
        log.close()


if __name__ == '__main__':
    main()
