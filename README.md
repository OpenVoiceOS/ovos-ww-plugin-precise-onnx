# ovos-ww-plugin-precise-onnx

A wake word plugin for [OpenVoiceOS](https://openvoiceos.org). It runs Precise-style wake word models with `onnxruntime` instead of `tflite`.

Download pre-trained models from [OpenVoiceOS/precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models).

---

## Install

```bash
pip install ovos_ww_plugin_precise_onnx
```

---

## Configuration

Add the plugin to the `hotwords` section of `mycroft.conf`.

```json
"listener": {
  "wake_word": "hey_mycroft"
},
"hotwords": {
  "hey_mycroft": {
    "module": "ovos-ww-plugin-precise-onnx",
    "model": "https://github.com/OpenVoiceOS/precise-lite-models/raw/master/wakewords/en/hey_mycroft.onnx",
    "trigger_level": 3,
    "sensitivity": 0.5
   }
}
```

Get community models from [OpenVoiceOS/precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models).

---

## Related projects

- [OpenVoiceOS/precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models): pre-trained and community wake word models for this plugin.
- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): loads and manages this plugin at runtime.
- [OpenVoiceOS/ovos-dinkum-listener](https://github.com/OpenVoiceOS/ovos-dinkum-listener): the listener service that uses wake word plugins like this one.

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
