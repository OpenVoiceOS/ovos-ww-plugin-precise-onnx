# precise-onnx plugin

## Description

OpenVoiceOS wake word plugin for precise using onnxruntime instead of tflite

download pre-trained [precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models)

## Configuration

Add the following to your hotwords section in mycroft.conf 

```json
"listener": {
  "wake_word": "hey mycroft"
},
"hotwords": {
  "hey mycroft": {
    "module": "ovos-ww-plugin-precise-onnx",
    "model": "https://github.com/OpenVoiceOS/precise-lite-models/raw/master/wakewords/en/hey_mycroft.onnx",
    "trigger_level": 3,
    "sensitivity": 0.5
   }
}
```

Get community models [here](https://github.com/OpenVoiceOS/precise-lite-models)

