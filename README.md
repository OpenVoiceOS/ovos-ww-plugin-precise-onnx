# precise-onnx plugin

---

OpenVoiceOS wake word plugin for precise using onnxruntime instead of tflite

download pre-trained [precise-lite-models](https://github.com/OpenVoiceOS/precise-lite-models)

---

## Install

```bash
pip install ovos_ww_plugin_precise_onnx
```

---

## Configuration

Add the following to your hotwords section in mycroft.conf 

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

Get community models [here](https://github.com/OpenVoiceOS/precise-lite-models)

---

## Credits

This work was made possible by the generous grant from [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS)

![](./ngi.png)

> This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) programme, under the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en) under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429). Additional funding is made available by the [Swiss State Secretariat for Education, Research and Innovation](https://www.sbfi.admin.ch/sbfi/en/home.html) (SERI).
