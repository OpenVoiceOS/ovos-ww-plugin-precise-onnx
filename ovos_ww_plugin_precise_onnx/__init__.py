from os import makedirs
from os.path import join, isfile, expanduser

import requests
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
import librosa
import numpy as np
import onnxruntime as ort


def vectorize_audio(audio, n_mfcc, n_features, rate, hop_length):
    """Compute MFCC features similar to precise_lite vectorize_raw()"""
    mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc.T[-n_features:, :]
    if mfcc.shape[0] < n_features:
        pad = np.zeros((n_features - mfcc.shape[0], n_mfcc))
        mfcc = np.vstack((pad, mfcc))
    return mfcc


class TriggerDetector:
    def __init__(self, chunk_size, sensitivity=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0

    def update(self, prob):
        chunk_activated = prob > 1.0 - self.sensitivity
        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or (chunk_activated and self.activation < 0):
                self.activation = -(8 * 2048) // self.chunk_size
            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False


class PreciseOnnxHotwordPlugin(HotWordEngine):

    def __init__(self, key_phrase="hey mycroft", config=None):
        super().__init__(key_phrase, config)
        self.has_found = False

        self.trigger_level = self.config.get('trigger_level', 3)
        self.sensitivity = self.config.get('sensitivity', 0.5)

        default_model = "https://github.com/OpenVoiceOS/precise-onnx-models/raw/master/wakewords/en/hey_mycroft.onnx"
        model = self.config.get('model', default_model)
        if model.startswith("http"):
            model = self.download_model(model)

        if not isfile(expanduser(model)):
            raise ValueError(f"Model not found: {model}")

        self.precise_model = expanduser(model)

        self.sample_rate = 16000
        chunk_ms = 20
        self.hop_length = int(self.sample_rate * chunk_ms / 1000)

        self.session = ort.InferenceSession(self.precise_model, providers=['CPUExecutionProvider'])
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape

        # Model expects (1, n_features, n_mfcc)
        self.n_features = int(input_shape[1])
        self.n_mfcc = int(input_shape[2])

        self.buffer = np.zeros(self.sample_rate, dtype=np.float32)
        self.detector = TriggerDetector(self.hop_length, self.sensitivity, self.trigger_level)


    @staticmethod
    def download_model(url):
        name = url.split("/")[-1]
        folder = join(xdg_data_home(), "precise-onnx")
        model_path = join(folder, name)
        if not isfile(model_path):
            LOG.info(f"Downloading ONNX model: {url}")
            response = requests.get(url)
            response.raise_for_status()
            makedirs(folder, exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(response.content)
            LOG.info(f"Model downloaded to {model_path}")
        return model_path

    def on_activation(self):
        self.has_found = True

    def update(self, chunk):
        chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.roll(self.buffer, -self.hop_length)
        self.buffer[-self.hop_length:] = chunk

        features = vectorize_audio(self.buffer, self.n_mfcc, self.n_features, self.sample_rate, self.hop_length)
        inp = features[np.newaxis, :, :].astype(np.float32)

        out = self.session.run(None, {self.input_name: inp})[0][0][0]
        self.has_found = self.detector.update(out)

    def found_wake_word(self):
        if self.has_found:
            self.has_found = False
            self.detector.activation = 0 # reset counter
            return True
        return False

