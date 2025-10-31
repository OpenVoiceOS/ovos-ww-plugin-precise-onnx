from os import makedirs
from os.path import join, isfile, expanduser

import requests
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
import numpy as np
from ovos_ww_plugin_precise_onnx.inference import PreciseOnnxEngine, TriggerDetector


class PreciseOnnxHotwordPlugin(HotWordEngine):

    def __init__(self, key_phrase="hey mycroft", config=None):
        """
        Initialize the hotword plugin, configure detection parameters, resolve or download the ONNX model, and create the PreciseOnnxEngine instance.
        
        Parameters:
            key_phrase (str): The wake word phrase to detect (e.g., "hey mycroft").
            config (dict or None): Optional configuration. Recognized keys:
                - 'trigger_level' (int): Number of trigger events required (default 3).
                - 'sensitivity' (float): Detection sensitivity threshold (default 0.5).
                - 'model' (str): Local path or HTTP URL to an ONNX model. If a URL is provided, the model will be downloaded to the user's data directory.
        
        Raises:
            ValueError: If the resolved model path does not point to an existing file.
        """
        super().__init__(key_phrase, config)
        self.trigger_flag = False  # Flag set when a trigger event is detected in 'update'
        self.trigger_level = self.config.get('trigger_level', 3)
        self.threshold = self.config.get('sensitivity', 0.5)

        default_model = "https://github.com/OpenVoiceOS/precise-onnx-models/raw/master/wakewords/en/hey_mycroft.onnx"
        model = self.config.get('model', default_model)
        if model.startswith("http"):
            model = self.download_model(model)

        if not isfile(expanduser(model)):
            raise ValueError(f"Model not found: {model}")

        self.precise_model = expanduser(model)
        self.engine = PreciseOnnxEngine(self.precise_model)


    @staticmethod
    def download_model(url):
        """
        Ensure an ONNX model from the given URL is available locally and return its path.
        
        Parameters:
            url (str): HTTP(S) URL of the ONNX model to fetch.
        
        Returns:
            model_path (str): Path to the local model file (downloaded if not already present).
        
        Raises:
            requests.HTTPError: If the HTTP request fails or returns a non-success status.
        """
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
        """
        Process a raw 16-bit PCM audio chunk in frames and update the internal wake-word trigger flag.
        
        Converts `chunk` (raw int16 PCM bytes) to a normalized float32 array in the range [-1.0, 1.0], iterates over successive frames of length `self.engine.hop_samples`, and for each complete frame asks the engine for a prediction; if the engine signals a detection, `self.trigger_flag` is set.
        
        Parameters:
            chunk (bytes): Raw audio bytes containing 16-bit signed PCM samples (int16).
        """
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.trigger_flag = self.engine.get_prediction(audio)

    def found_wake_word(self):
        """
        Check whether a wake word was detected since the last call and reset the detector state.
        
        This resets the internal trigger flag and clears the underlying engine state when a pending wake-word event is consumed.
        
        Returns:
            bool: `True` if a wake word was detected since the last call, `False` otherwise.
        """
        if self.trigger_flag:
            # Wake word was found, reset the flag for the next check
            self.trigger_flag = False
            self.engine.clear()
            return True
        return False