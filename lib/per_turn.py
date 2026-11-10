"""
Per-turn language detection for mixed-language recordings (nextcloud/stt_whisper2#73).

Whisper detects the language once (first ~30 s) and keeps it for the whole file.
This module:
  1. splits the audio at pauses with the bundled Silero VAD,
  2. detects the language of each speech chunk >= MIN_DETECT_S
     (accepted at probability >= MIN_PROB; short/uncertain chunks inherit
     the previous chunk's language),
  3. merges neighbouring chunks with the same language into spans,
  4. transcribes each span with language= fixed,
  5. shifts segment (and word) timestamps back to recording time.

The return value has the same shape as WhisperModel.transcribe():
(lazy segment generator, TranscriptionInfo), so text/SRT/VTT builders are unchanged.
A single-language file becomes one span that covers the whole recording.
"""

import dataclasses
import logging
import os
from typing import Callable, Iterator, Optional

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

LOGGER = logging.getLogger(os.environ.get("APP_ID", "stt_whisper2") + ".per_turn")

SAMPLING_RATE = 16000

PER_TURN_LANGUAGE = os.environ.get("STT_WHISPER2_PER_TURN_LANGUAGE", "1") == "1"
try:
    MIN_SILENCE_MS = int(os.environ.get("STT_WHISPER2_PER_TURN_MIN_SILENCE_MS", "1000"))
    MIN_DETECT_S = float(os.environ.get("STT_WHISPER2_PER_TURN_MIN_DETECT_S", "3"))
    MIN_PROB = float(os.environ.get("STT_WHISPER2_PER_TURN_MIN_PROB", "0.6"))
except ValueError:
    raise Exception("Failed to parse per-turn language settings. All values must be valid numbers")


def detect_spans(model, audio, vad_threshold: float = 0.5,
                 min_silence_ms: int = MIN_SILENCE_MS,
                 min_detect_s: float = MIN_DETECT_S,
                 min_prob: float = MIN_PROB,
                 on_progress: Optional[Callable[[float], None]] = None) -> list[dict]:
    """Return spans [{'start': sample, 'end': sample, 'language': str|None}] covering the whole audio.

    on_progress, if given, is called with the fraction (0..1) of the audio scanned so far.
    """
    total = len(audio)
    chunks = get_speech_timestamps(
        audio,
        VadOptions(threshold=vad_threshold, min_silence_duration_ms=min_silence_ms),
    )
    if not chunks:
        return [{"start": 0, "end": total, "language": None}]

    # 2. language per chunk
    langs: list[Optional[str]] = []
    prev = None
    for c in chunks:
        dur = (c["end"] - c["start"]) / SAMPLING_RATE
        lang = None
        if dur >= min_detect_s:
            detected, prob, _ = model.detect_language(audio=audio[c["start"]:c["end"]])
            LOGGER.debug(f"chunk {c['start']/SAMPLING_RATE:.1f}-{c['end']/SAMPLING_RATE:.1f}s: {detected} p={prob:.2f}")
            if prob >= min_prob:
                lang = detected
            if on_progress is not None:
                on_progress(c["end"] / total)
        if lang is None:
            lang = prev
        langs.append(lang)
        if lang is not None:
            prev = lang

    # leading chunks with no confident language take the first known one
    first_known = next((l for l in langs if l is not None), None)
    langs = [l if l is not None else first_known for l in langs]

    # 3. merge neighbours; span boundaries sit in the middle of the pauses,
    # so no audio (including VAD padding) is lost between spans
    spans = [{"start": 0, "end": total, "language": langs[0]}]
    for i in range(1, len(chunks)):
        if langs[i] != spans[-1]["language"]:
            boundary = (chunks[i - 1]["end"] + chunks[i]["start"]) // 2
            spans[-1]["end"] = boundary
            spans.append({"start": boundary, "end": total, "language": langs[i]})
    return spans


def _shift(segment, offset: float, new_id: int):
    words = segment.words
    if words:
        words = [dataclasses.replace(w, start=w.start + offset, end=w.end + offset) for w in words]
    return dataclasses.replace(
        segment, id=new_id, start=segment.start + offset, end=segment.end + offset, words=words
    )


def transcribe_per_turn(model, file_name, vad_filter: bool = True,
                        vad_parameters: Optional[dict] = None,
                        on_progress: Optional[Callable[[float], None]] = None, **kwargs):
    """Drop-in replacement for model.transcribe(file_name, vad_filter=..., vad_parameters=...).

    on_progress reports the language detection phase, which runs before any segment is produced.
    """
    audio = decode_audio(file_name, sampling_rate=SAMPLING_RATE)
    vad_threshold = (vad_parameters or {}).get("threshold", 0.5)
    spans = detect_spans(model, audio, vad_threshold=vad_threshold, on_progress=on_progress)
    LOGGER.info("per-turn spans: " + ", ".join(
        f"{s['start']/SAMPLING_RATE:.1f}-{s['end']/SAMPLING_RATE:.1f}s={s['language']}" for s in spans
    ))

    # all spans unknown -> plain behaviour
    if len(spans) == 1 and spans[0]["language"] is None:
        return model.transcribe(audio, vad_filter=vad_filter, vad_parameters=vad_parameters, **kwargs)

    # Start span 0 now to get a TranscriptionInfo to return; segments stay lazy,
    # so the callers' per-segment progress reporting keeps working.
    def run(span):
        return model.transcribe(
            audio[span["start"]:span["end"]],
            language=span["language"],
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            **kwargs,
        )

    first_segments, first_info = run(spans[0])
    info = dataclasses.replace(
        first_info,
        duration=len(audio) / SAMPLING_RATE,
    )

    def gen() -> Iterator:
        seg_id = 0
        for k, span in enumerate(spans):
            segments = first_segments if k == 0 else run(span)[0]
            offset = span["start"] / SAMPLING_RATE
            for seg in segments:
                seg_id += 1
                yield _shift(seg, offset, seg_id)

    return gen(), info
