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

        # Process audio frame by frame
        for start in range(0, len(audio), self.engine.hop_samples):
            frame = audio[start:start + self.engine.hop_samples]
            if len(frame) < self.engine.hop_samples:
                continue
            self.trigger_flag = self.engine.get_prediction(frame)

    def found_wake_word(self):
        """
        Returns True if a trigger was detected since the last call
        and resets the trigger flag.
        """
        if self.trigger_flag:
            # Wake word was found, reset the flag for the next check
            self.trigger_flag = False
            self.engine.clear()
            return True
        return False
