"""
Unit tests for lib/per_turn.py (per-turn language detection, issue #73).

No Whisper model is needed: the model is a fake, and VAD is either stubbed
or (in the last test) the real bundled Silero VAD on test/res/jfk.wav.

    pip install -r requirements.txt pytest
    pytest test/test_per_turn.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from faster_whisper.transcribe import Segment, TranscriptionInfo, Word

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import per_turn  # noqa: E402

SR = per_turn.SAMPLING_RATE
JFK = Path(__file__).resolve().parent / "res" / "jfk.wav"


class FakeModel:
    """Detects languages from a {chunk_duration_s: (lang, prob)} table and records transcribe() calls."""

    def __init__(self, by_duration=None):
        self.by_duration = by_duration or {}
        self.detect_calls = []
        self.transcribe_calls = []

    def detect_language(self, audio):
        dur = round(len(audio) / SR, 2)
        self.detect_calls.append(dur)
        lang, prob = self.by_duration.get(dur, ("xx", 0.0))
        return lang, prob, []

    def transcribe(self, audio, language=None, **kwargs):
        dur = len(audio) / SR
        self.transcribe_calls.append((round(dur, 2), language))
        seg = Segment(
            id=1, seek=0, start=1.0, end=2.0, text=f" [{language}]", tokens=[],
            avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0,
            words=[Word(start=1.0, end=1.5, word="x", probability=1.0)], temperature=0.0,
        )
        info = TranscriptionInfo(
            language=language or "en", language_probability=1.0, duration=dur,
            duration_after_vad=dur, all_language_probs=None,
            transcription_options=None, vad_options=None,
        )
        return iter([seg]), info


@pytest.fixture
def stub_audio(monkeypatch):
    """Silent audio of a given length with VAD stubbed to return the given (start_s, end_s) chunks."""

    def make(total_s, chunks_s):
        audio = np.zeros(int(total_s * SR), dtype=np.float32)
        monkeypatch.setattr(
            per_turn, "get_speech_timestamps",
            lambda a, opts: [{"start": int(s * SR), "end": int(e * SR)} for s, e in chunks_s],
        )
        monkeypatch.setattr(per_turn, "decode_audio", lambda f, sampling_rate: audio)
        return audio

    return make


def langs(spans):
    return [s["language"] for s in spans]


def bounds_s(spans):
    return [(s["start"] / SR, s["end"] / SR) for s in spans]


# --- detect_spans -------------------------------------------------------------

def test_issue_scenario_es_en_es(stub_audio):
    audio = stub_audio(84, [(0, 33), (34.5, 64.5), (66, 84)])
    model = FakeModel({33.0: ("es", 0.95), 30.0: ("en", 0.97), 18.0: ("es", 0.9)})
    spans = per_turn.detect_spans(model, audio)
    assert langs(spans) == ["es", "en", "es"]
    # boundaries in the middle of the pauses, spans cover the whole file without gaps
    assert bounds_s(spans) == [(0, 33.75), (33.75, 65.25), (65.25, 84)]


def test_single_language_is_one_span_over_whole_file(stub_audio):
    audio = stub_audio(40, [(1, 12), (14, 25), (27, 39)])
    model = FakeModel({11.0: ("de", 0.9), 12.0: ("de", 0.9)})
    spans = per_turn.detect_spans(model, audio)
    assert spans == [{"start": 0, "end": len(audio), "language": "de"}]


def test_short_chunk_inherits_previous_language_without_detection(stub_audio):
    audio = stub_audio(30, [(0, 10), (11, 12), (13, 30)])  # 1 s "ok" between two English turns
    model = FakeModel({10.0: ("en", 0.9), 17.0: ("en", 0.9)})
    spans = per_turn.detect_spans(model, audio)
    assert langs(spans) == ["en"]
    assert 1.0 not in model.detect_calls  # the short chunk was never sent to detect_language


def test_uncertain_chunk_inherits_previous_language(stub_audio):
    audio = stub_audio(30, [(0, 10), (11, 20), (21, 30)])
    model = FakeModel({10.0: ("de", 0.9), 9.0: ("en", 0.4)})  # 2nd and 3rd chunk are 9 s, p=0.4
    spans = per_turn.detect_spans(model, audio)
    assert langs(spans) == ["de"]


def test_leading_uncertain_chunk_takes_first_confident_language(stub_audio):
    audio = stub_audio(20, [(0, 2), (3, 20)])  # 2 s greeting, then a long Spanish turn
    model = FakeModel({17.0: ("es", 0.9)})
    spans = per_turn.detect_spans(model, audio)
    assert spans == [{"start": 0, "end": len(audio), "language": "es"}]


def test_no_confident_language_gives_one_unknown_span(stub_audio):
    audio = stub_audio(20, [(0, 9), (10, 19)])
    model = FakeModel({9.0: ("en", 0.3)})
    assert per_turn.detect_spans(model, audio) == [{"start": 0, "end": len(audio), "language": None}]


def test_no_speech_gives_one_unknown_span(stub_audio):
    audio = stub_audio(5, [])
    assert per_turn.detect_spans(FakeModel(), audio) == [{"start": 0, "end": len(audio), "language": None}]


def test_thresholds_are_respected(stub_audio):
    audio = stub_audio(12, [(0, 5), (6, 12)])
    model = FakeModel({5.0: ("en", 0.7), 6.0: ("de", 0.7)})
    assert langs(per_turn.detect_spans(model, audio)) == ["en", "de"]
    assert langs(per_turn.detect_spans(model, audio, min_prob=0.8)) == [None]
    assert langs(per_turn.detect_spans(model, audio, min_detect_s=5.5)) == ["de"]


def test_detection_reports_progress(stub_audio):
    audio = stub_audio(40, [(0, 10), (12, 13), (15, 40)])
    model = FakeModel({10.0: ("en", 0.9), 25.0: ("en", 0.9)})
    progress = []
    per_turn.detect_spans(model, audio, on_progress=progress.append)
    # reported once per detected chunk (the 1 s chunk is too short), up to the end of the last chunk
    assert progress == [0.25, 1.0]


# --- transcribe_per_turn ------------------------------------------------------

def test_segments_are_shifted_to_recording_time_and_renumbered(stub_audio):
    stub_audio(84, [(0, 33), (34.5, 64.5), (66, 84)])
    model = FakeModel({33.0: ("es", 0.95), 30.0: ("en", 0.97), 18.0: ("es", 0.9)})
    segments, info = per_turn.transcribe_per_turn(model, "ignored.wav")
    segments = list(segments)

    assert [s.text for s in segments] == [" [es]", " [en]", " [es]"]
    assert [s.id for s in segments] == [1, 2, 3]
    # fake segments start 1 s into each span
    assert [s.start for s in segments] == [1.0, 34.75, 66.25]
    assert [s.words[0].start for s in segments] == [1.0, 34.75, 66.25]
    # duration is the full recording, so progress = segment.start / duration stays correct
    assert info.duration == 84
    assert model.transcribe_calls == [(33.75, "es"), (31.5, "en"), (18.75, "es")]


def test_vad_settings_are_passed_through(stub_audio):
    stub_audio(10, [(0, 10)])
    seen = {}

    class Recorder(FakeModel):
        def transcribe(self, audio, language=None, **kwargs):
            seen.update(kwargs)
            return super().transcribe(audio, language=language)

    params = {"threshold": 0.5, "min_silence_duration_ms": 2000}
    segments, _ = per_turn.transcribe_per_turn(
        Recorder({10.0: ("en", 0.9)}), "x.wav", vad_filter=True, vad_parameters=params
    )
    list(segments)
    assert seen == {"vad_filter": True, "vad_parameters": params}


def test_falls_back_to_plain_transcribe_without_language(stub_audio):
    stub_audio(20, [(0, 9), (10, 19)])
    model = FakeModel({9.0: ("en", 0.3)})
    segments, _ = per_turn.transcribe_per_turn(model, "x.wav")
    list(segments)
    assert model.transcribe_calls == [(20.0, None)]  # one call, whole file, auto language


# --- real Silero VAD ------------------------------------------------------------

@pytest.mark.skipif(not JFK.exists(), reason="test/res/jfk.wav missing")
def test_real_vad_single_language_file_is_one_span():
    audio = per_turn.decode_audio(str(JFK), sampling_rate=SR)

    class AlwaysEnglish(FakeModel):
        def detect_language(self, audio):
            return "en", 0.99, []

    spans = per_turn.detect_spans(AlwaysEnglish(), audio)
    assert spans == [{"start": 0, "end": len(audio), "language": "en"}]
