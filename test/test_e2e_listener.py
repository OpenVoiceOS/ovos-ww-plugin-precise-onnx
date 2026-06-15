"""
End-to-end ovoscope listener test for ovos-ww-plugin-precise-onnx.

Tests:
  - Engine loads and processes audio without error.
  - Negative: command.wav (non-wakeword speech) does NOT trigger detection.
  - Positive: hey_mycroft.wav triggers the model (hard assertion).
"""
import wave
from pathlib import Path

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed")
plugin_mod = pytest.importorskip(
    "ovos_ww_plugin_precise_onnx", reason="ovos-ww-plugin-precise-onnx not installed"
)

from ovoscope.listener import get_mini_listener  # noqa: E402
from ovos_ww_plugin_precise_onnx import PreciseOnnxHotwordPlugin  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_wav_pcm(path: Path) -> bytes:
    """Return raw s16le PCM bytes from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _make_engine() -> PreciseOnnxHotwordPlugin:
    """Instantiate the plugin, downloading the default hey_mycroft model on first use."""
    return PreciseOnnxHotwordPlugin(
        key_phrase="hey mycroft",
        config={"trigger_level": 3, "sensitivity": 0.5},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Shared engine instance (loading ONNX once per module)."""
    return _make_engine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_loads():
    """Engine instantiates without error, downloading the default model."""
    eng = _make_engine()
    assert eng is not None
    assert eng.engine is not None


def test_negative_detection_command_wav(engine):
    """command.wav (non-wakeword speech) must NOT trigger the wakeword engine."""
    wav_path = FIXTURES / "command.wav"
    assert wav_path.exists(), f"Negative fixture missing: {wav_path}"

    listener = get_mini_listener(ww_instances={"hey_mycroft": engine})

    pcm = _read_wav_pcm(wav_path)
    detected, frame_idx = listener.scan_for_wakeword(pcm, frame_size=2048)

    listener.shutdown()
    # Reset engine state for subsequent tests
    engine.engine.clear()
    engine.trigger_flag = False

    assert not detected, (
        f"Wakeword falsely triggered on command.wav at frame {frame_idx}"
    )


def test_positive_detection_hey_mycroft(engine):
    """hey_mycroft.wav must trigger the wakeword engine (hard assertion).

    The fixture is a real Microsoft edge-tts en-US-AriaNeural rendering of
    'hey mycroft' with 0.5 s of leading silence prepended, resampled to
    16 kHz / mono / s16le.  The precise-onnx model fires on it at default
    threshold (sensitivity=0.5, trigger_level=3).
    """
    wav_path = FIXTURES / "hey_mycroft.wav"
    assert wav_path.exists(), f"Positive fixture missing: {wav_path}"

    # Fresh engine state
    engine.engine.clear()
    engine.trigger_flag = False

    pcm = _read_wav_pcm(wav_path)
    # hop_samples * 2 bytes-per-int16 = one hop worth of raw PCM bytes
    frame_size = engine.engine.hop_samples * 2

    detected = False
    frames = [pcm[i: i + frame_size] for i in range(0, len(pcm), frame_size)]
    for chunk in frames:
        if not chunk:
            continue
        engine.update(chunk)
        if engine.found_wake_word():
            detected = True
            break

    assert detected, (
        "PreciseOnnxHotwordPlugin did NOT detect the wakeword in hey_mycroft.wav. "
        "The fixture is test/fixtures/hey_mycroft.wav (16 kHz mono s16le). "
        "Check audio format, model path, or threshold config."
    )
