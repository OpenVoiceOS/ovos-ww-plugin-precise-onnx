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


class PreciseOnnxHotwordPlugin(HotWordEngine):

    def __init__(self, key_phrase="hey mycroft", config=None):
        super().__init__(key_phrase, config)

        self.activations = 0
        self.trigger_level = self.config.get('trigger_level', 1)
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

    def update(self, chunk):
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        for start in range(0, len(audio), self.hop_length):
            frame = audio[start:start + self.hop_length]
            if len(frame) < self.hop_length:
                continue
            self.buffer = np.roll(self.buffer, -self.hop_length)
            self.buffer[-self.hop_length:] = frame

            features = vectorize_audio(self.buffer, self.n_mfcc, self.n_features,
                                       self.sample_rate, self.hop_length)
            inp = features[np.newaxis, :, :].astype(np.float32)
            out = self.session.run(None, {self.input_name: inp})[0][0][0]
            if out > 0:
                self.activations += 1

    def found_wake_word(self):
        if self.activations >= self.trigger_level:
            self.activations = 0  # reset counter
            return True
        return False
