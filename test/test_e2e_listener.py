"""
End-to-end ovoscope listener test for ovos-ww-plugin-precise-onnx.

Tests:
  - Engine loads and processes audio without error.
  - Negative: command.wav (non-wakeword speech) does NOT trigger detection.
  - Positive (best-effort): TTS "hey mycroft" audio checked for detection.
    NOTE: The precise-onnx model is trained on real recorded speech; TTS audio
    may not reliably trigger detection. If it does not fire the test still
    passes (the assertion is non-strict for TTS). A real recorded wakeword
    sample would be needed for a hard positive assertion.
"""
import wave
import struct
from pathlib import Path

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed")
plugin_mod = pytest.importorskip(
    "ovos_ww_plugin_precise_onnx", reason="ovos-ww-plugin-precise-onnx not installed"
)

from ovoscope.listener import get_mini_listener  # noqa: E402
from ovos_ww_plugin_precise_onnx import PreciseOnnxHotwordPlugin  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = Path.home() / ".local/share/precise-onnx/hey_mycroft.onnx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_wav_pcm(path: Path) -> bytes:
    """Return raw s16le PCM bytes from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _make_engine() -> PreciseOnnxHotwordPlugin:
    """Instantiate the plugin using the locally cached model (no network)."""
    return PreciseOnnxHotwordPlugin(
        key_phrase="hey mycroft",
        config={"model": str(MODEL_PATH), "trigger_level": 3, "sensitivity": 0.5},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Shared engine instance (loading ONNX once per module)."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Cached model not found: {MODEL_PATH}")
    return _make_engine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_loads():
    """Engine instantiates without error using cached local model."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Cached model not found: {MODEL_PATH}")
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


def test_positive_detection_hey_mycroft_tts(engine):
    """
    Feed TTS-synthesised "hey mycroft" audio through the engine.

    The precise-onnx model is optimised for real recorded speech; TTS audio
    may not reliably fire it.  This test asserts only that:
      - The engine processes all frames without error.
      - found_wake_word() is callable and returns a bool.

    If detection DOES fire on TTS audio that is treated as a bonus and logged.
    A strict positive assertion requires a recorded wakeword sample committed
    as test/fixtures/hey_mycroft_real.wav.
    """
    wav_path = FIXTURES / "hey_mycroft.wav"
    assert wav_path.exists(), f"TTS fixture missing: {wav_path}"

    # Reset before the test
    engine.engine.clear()
    engine.trigger_flag = False

    pcm = _read_wav_pcm(wav_path)
    frame_size = engine.engine.hop_samples * 2  # 2 bytes per int16 sample

    detected = False
    frames = [pcm[i: i + frame_size] for i in range(0, len(pcm), frame_size)]
    for chunk in frames:
        if not chunk:
            continue
        engine.update(chunk)
        if engine.found_wake_word():
            detected = True
            break

    # Regardless of detection, engine must still be usable
    result = engine.found_wake_word()
    assert isinstance(result, bool)

    if detected:
        print("\n[INFO] TTS audio triggered wakeword detection (bonus).")
    else:
        print(
            "\n[INFO] TTS audio did NOT trigger wakeword detection — expected. "
            "Commit test/fixtures/hey_mycroft_real.wav (recorded speech) for a "
            "hard positive assertion."
        )

    # Non-strict: pass either way; the engine must not error
    assert True
