from math import exp, log, sqrt, pi, floor
from typing import Tuple

import numpy as np
import onnxruntime as ort
from sonopy import mfcc_spec


class ThresholdDecoder:
    """
    Decode raw network output into a relatively linear threshold using
    This works by estimating the logit normal distribution of network
    activations using a series of averages and standard deviations to
    calculate a cumulative probability distribution
    """

    def __init__(self, mu_stds: Tuple[Tuple[float, float]], center=0.5,
                 resolution=200, min_z=-4, max_z=4):
        self.min_out = int(min(mu + min_z * std for mu, std in mu_stds))
        self.max_out = int(max(mu + max_z * std for mu, std in mu_stds))
        self.out_range = self.max_out - self.min_out
        self.cd = np.cumsum(self._calc_pd(mu_stds, resolution))
        self.center = center

    @staticmethod
    def sigmoid(x):
        """Sigmoid squashing function for scalars"""
        return 1 / (1 + exp(-x))

    @staticmethod
    def asigmoid(x):
        """Inverse sigmoid (logit) for scalars"""
        return -log(1 / x - 1)

    @staticmethod
    def pdf(x, mu, std):
        """Probability density function (normal distribution)"""
        if std == 0:
            return 0
        return (1.0 / (std * sqrt(2 * pi))) * np.exp(
            -(x - mu) ** 2 / (2 * std ** 2))

    def decode(self, raw_output: float) -> float:
        if raw_output == 1.0 or raw_output == 0.0:
            return raw_output
        if self.out_range == 0:
            cp = int(raw_output > self.min_out)
        else:
            ratio = (self.asigmoid(raw_output) - self.min_out) / self.out_range
            ratio = min(max(ratio, 0.0), 1.0)
            cp = self.cd[int(ratio * (len(self.cd) - 1) + 0.5)]
        if cp < self.center:
            return 0.5 * cp / self.center
        else:
            return 0.5 + 0.5 * (cp - self.center) / (1 - self.center)

    def encode(self, threshold: float) -> float:
        threshold = 0.5 * threshold / self.center
        if threshold < 0.5:
            cp = threshold * self.center * 2
        else:
            cp = (threshold - 0.5) * 2 * (1 - self.center) + self.center
        ratio = np.searchsorted(self.cd, cp) / len(self.cd)
        return self.sigmoid(self.min_out + self.out_range * ratio)

    def _calc_pd(self, mu_stds, resolution):
        points = np.linspace(self.min_out, self.max_out,
                             resolution * self.out_range)
        return np.sum([self.pdf(points, mu, std) for mu, std in mu_stds],
                      axis=0) / (resolution * len(mu_stds))


class TriggerDetector:
    """
    Reads predictions and detects activations
    This prevents multiple close activations from occurring when
    the predictions look like ...!!!..!!...
    """

    def __init__(self, chunk_size, sensitivity=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0

    def update(self, prob: float) -> bool:
        """Returns whether the new prediction caused an activation"""
        chunk_activated = prob > 1.0 - self.sensitivity
        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or chunk_activated and self.activation < 0:
                self.activation = -(8 * 2048) // self.chunk_size
            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False


class PreciseOnnxEngine:
    """Listener that preprocesses audio into MFCC vectors
     and executes neural networks"""

    def __init__(self, model_path: str, sample_rate: int = 16000, threshold=0.5, trigger_level=3):

        self.sample_rate = sample_rate

        # values taken from original precise code
        self.n_mfcc: int = 13
        self.n_filt: int = 20
        self.n_fft: int = 512
        self.threshold_config: tuple = ((6, 4),)
        self.threshold_center: float = 0.2
        self.hop_t = 0.05
        self.window_t = 0.1
        self.buffer_t: float = 1.5

        self.threshold_decoder = ThresholdDecoder(self.threshold_config, self.threshold_center)
        self.trigger_detector = TriggerDetector(chunk_size=2048,
                                                sensitivity=threshold,
                                                trigger_level=trigger_level)

        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.n_features, self.n_mfcc))
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape

        # Model expects (1, n_features, n_mfcc)
        n_features = int(input_shape[1])
        n_mfcc = int(input_shape[2])
        if n_features != self.n_features or n_mfcc != self.n_mfcc:
            raise ValueError(f"Invalid onnx model input_shape=({n_features}, {n_mfcc})  [expected=({self.n_features}, {self.n_mfcc})]")

    @property
    def buffer_samples(self):
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

    @property
    def window_samples(self):
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        return int(self.buffer_t * self.sample_rate)

    def clear(self):
        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.n_features, self.n_mfcc))

    def _update_vectors(self, buffer_audio: np.ndarray):

        self.window_audio = np.concatenate((self.window_audio, buffer_audio))

        if len(self.window_audio) >= self.window_samples:

            new_features = mfcc_spec(
                self.window_audio, self.sample_rate, (self.window_samples, self.hop_samples),
                num_filt=self.n_filt, fft_size=self.n_fft, num_coeffs=self.n_mfcc
            )
            self.window_audio = self.window_audio[
                len(new_features) * self.hop_samples:]
            if len(new_features) > len(self.mfccs):
                new_features = new_features[-len(self.mfccs):]
            self.mfccs = np.concatenate(
                (self.mfccs[len(new_features):], new_features))

        return self.mfccs

    def update(self, stream: np.ndarray) -> float:
        mfccs = self._update_vectors(stream)
        input_data = mfccs[np.newaxis, :, :].astype(np.float32)  # add batch dim
        raw_output = self.session.run(None, {self.input_name: input_data})[0][0][0]
        return self.threshold_decoder.decode(raw_output)

    def get_prediction(self, chunk) -> bool:
        prob = self.update(chunk)
        return self.trigger_detector.update(prob)
