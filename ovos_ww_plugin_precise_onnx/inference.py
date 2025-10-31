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
        """
        Initialize the ThresholdDecoder by building an output range and cumulative distribution from provided mean/std pairs.
        
        Parameters:
            mu_stds (Tuple[Tuple[float, float]]): Iterable of (mean, std) pairs describing logit-normal component distributions.
            center (float): Center value used for piecewise scaling of decoded probabilities.
            resolution (int): Number of points used to discretize the output range when computing the probability density.
            min_z (float): Minimum z-score multiplier applied to each mean/std to determine the lower bound of the output range.
            max_z (float): Maximum z-score multiplier applied to each mean/std to determine the upper bound of the output range.
        
        Notes:
            - Computes min_out and max_out from mu_stds using min_z/max_z, stores their integer bounds and range.
            - Precomputes a cumulative distribution array (`cd`) by evaluating the combined probability density across the discretized range.
        """
        self.min_out = int(min(mu + min_z * std for mu, std in mu_stds))
        self.max_out = int(max(mu + max_z * std for mu, std in mu_stds))
        self.out_range = self.max_out - self.min_out
        self.cd = np.cumsum(self._calc_pd(mu_stds, resolution))
        self.center = center

    @staticmethod
    def sigmoid(x):
        """
        Compute the logistic sigmoid of x.
        
        Parameters:
            x (float): Input value.
        
        Returns:
            float: Value in the range 0 to 1 equal to 1 / (1 + exp(-x)).
        """
        return 1 / (1 + exp(-x))

    @staticmethod
    def asigmoid(x):
        """
        Compute the logit (inverse of the logistic sigmoid) of a probability.
        
        Parameters:
            x (float): A probability strictly greater than 0 and strictly less than 1.
        
        Returns:
            float: The logit value, computed as ln(x / (1 - x)).
        """
        return -log(1 / x - 1)

    @staticmethod
    def pdf(x, mu, std):
        """
        Evaluate the probability density of a normal (Gaussian) distribution at a given point.
        
        Parameters:
            x (float): Point at which to evaluate the density.
            mu (float): Mean of the normal distribution.
            std (float): Standard deviation of the normal distribution.
        
        Returns:
            float: The probability density at `x` for N(mu, std). If `std` is 0, returns 0.
        """
        if std == 0:
            return 0
        return (1.0 / (std * sqrt(2 * pi))) * np.exp(
            -(x - mu) ** 2 / (2 * std ** 2))

    def decode(self, raw_output: float) -> float:
        """
        Map a model's sigmoid-like scalar output into a centered, piecewise-linear calibrated probability.
        
        Interprets `raw_output` (expected in [0, 1]) and converts it into a probability-like value in [0, 1] using the decoder's internal cumulative distribution and center point. If `raw_output` is exactly 0.0 or 1.0, it is returned unchanged. When the decoder's output range is zero, the function performs a simple threshold comparison against the minimum modeled output. Otherwise the input is inverted through the decoder's logit transform, normalized into the modeled output range, and looked up in the precomputed cumulative distribution. The resulting cumulative probability is then remapped with a piecewise linear transform that compresses values below `center` into [0, 0.5) and values at or above `center` into [0.5, 1.0].
        
        Parameters:
            raw_output (float): The model output to decode, typically in the range [0, 1].
        
        Returns:
            float: A calibrated probability in [0, 1], with values below `center` mapped into [0, 0.5) and values at or above `center` mapped into [0.5, 1.0]; exact 0.0 and 1.0 inputs are returned unchanged.
        """
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
        """
        Map a threshold value into a model output in [0, 1] that corresponds to the decoder's internal cumulative distribution.
        
        Parameters:
            threshold (float): Desired threshold value (probability-like) to encode.
        
        Returns:
            float: A sigmoid-squashed model output in the range [0, 1] that represents the given threshold.
        """
        threshold = 0.5 * threshold / self.center
        if threshold < 0.5:
            cp = threshold * self.center * 2
        else:
            cp = (threshold - 0.5) * 2 * (1 - self.center) + self.center
        ratio = np.searchsorted(self.cd, cp) / len(self.cd)
        return self.sigmoid(self.min_out + self.out_range * ratio)

    def _calc_pd(self, mu_stds, resolution):
        """
        Compute a normalized combined probability density over the decoder's output range from multiple Gaussian components.
        
        Parameters:
            mu_stds (iterable of tuple(float, float)): Sequence of (mean, std) pairs defining each Gaussian component.
            resolution (int): Number of density samples per unit of output range; the returned array length equals resolution * self.out_range.
        
        Returns:
            numpy.ndarray: 1-D array of probability density values evaluated across the decoder's output span [self.min_out, self.max_out], normalized so that the sum of densities corresponds to the chosen sampling resolution and number of components.
        """
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
        """
        Initialize a TriggerDetector with parameters controlling sensitivity and debounce.
        
        Parameters:
            chunk_size (int): Number of samples (or units) processed per update call; used to align detection timing.
            sensitivity (float): Relative proximity threshold (0.0–1.0) used to decide whether an incoming probability is considered close enough to activation.
            trigger_level (int): Number of consecutive qualifying updates required to emit a positive activation.
        
        Attributes:
            activation (int): Internal counter tracking consecutive positive (or negative) updates; initialized to 0.
        """
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0

    def update(self, prob: float) -> bool:
        """
        Update the detector with a new probability and indicate whether it produced an activation.
        
        Parameters:
            prob (float): Probability for the current chunk, expected in the range [0, 1].
        
        Returns:
            bool: `True` if an activation event is produced (activation count exceeded the trigger level), `False` otherwise.
        """
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

        """
        Initialize a PreciseOnnxEngine configured for streaming MFCC extraction and ONNX inference.
        
        Sets up audio preprocessing parameters (MFCC/window/hop/buffer), creates a ThresholdDecoder and TriggerDetector, allocates internal audio and feature buffers, and loads and validates the ONNX inference session for the provided model.
        
        Parameters:
            model_path (str): Filesystem path to the ONNX model used for inference.
            sample_rate (int): Audio sample rate in Hz used for feature extraction.
            threshold (float): Sensitivity passed to the TriggerDetector to control activation gating.
            trigger_level (int): Number of consecutive trigger increments required to emit an activation.
        
        Raises:
            ValueError: If the model's input shape does not match the expected (n_features, n_mfcc).
        """
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
        """
        Compute the number of buffered audio samples aligned down to the nearest hop boundary.
        
        Returns:
            int: Number of samples in the buffer, rounded down to a multiple of the hop size.
        """
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        """
        Number of feature vectors available given the current buffer, window, and hop sizes.
        
        Returns:
            int: Count of consecutive analysis windows (feature vectors) that can be produced from the buffered audio.
        """
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

    @property
    def window_samples(self):
        """
        Compute the number of audio samples contained in a single analysis window.
        
        Returns:
            int: The window size in samples, calculated as sample_rate * window_t and rounded to the nearest integer.
        """
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        """
        Compute the hop (stride) size in samples for the analysis windows.
        
        Returns:
            hop_samples (int): Number of samples corresponding to the configured hop time, rounded to the nearest integer.
        """
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        """
        Compute the maximum number of audio samples the internal buffer can hold.
        
        Returns:
            max_samples (int): The buffer capacity in samples, computed as buffer_t multiplied by sample_rate.
        """
        return int(self.buffer_t * self.sample_rate)

    def clear(self):
        """
        Reset the audio and feature buffers to their initial empty states.
        
        This clears the internal rolling audio window and zeroes the MFCC feature matrix so subsequent processing starts from a clean buffer.
        """
        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.n_features, self.n_mfcc))

    def _update_vectors(self, buffer_audio: np.ndarray):

        """
        Update the internal audio window with new samples and return the current stacked MFCC feature matrix.
        
        Appends the provided audio chunk to the internal window buffer; when enough samples are available for one or more analysis windows, computes MFCC feature vectors for those windows, advances the audio window by the corresponding hop amount, and updates the rolling MFCC buffer to keep only the most recent feature frames.
        
        Parameters:
            buffer_audio (np.ndarray): 1-D array of new audio samples to append to the internal window.
        
        Returns:
            np.ndarray: The current MFCC feature matrix with shape (n_features, n_mfcc), containing the most recent stacked feature frames.
        """
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
        """
        Process an incoming audio chunk to produce a calibrated detection probability.
        
        Parameters:
            stream (np.ndarray): 1-D audio samples for the current chunk.
        
        Returns:
            float: Calibrated probability for the model's detection for this chunk, scaled according to the configured threshold decoder.
        """
        mfccs = self._update_vectors(stream)
        input_data = mfccs[np.newaxis, :, :].astype(np.float32)  # add batch dim
        raw_output = self.session.run(None, {self.input_name: input_data})[0][0][0]
        return self.threshold_decoder.decode(raw_output)

    def get_prediction(self, chunk) -> bool:
        """
        Return whether the provided audio chunk produces a trigger activation.
        
        Parameters:
            chunk (ndarray): A 1-D numpy array of audio samples for the current input chunk.
        
        Returns:
            bool: `True` if the detector reports an activation for this chunk, `False` otherwise.
        """
        prob = self.update(chunk)
        return self.trigger_detector.update(prob)
