# 제3자 구성요소와 라이선스

이 저장소에는 모델 파일이 들어 있지 않습니다. 앱을 열면 사용자의 브라우저가 아래 구성요소를 직접 내려받습니다.

| 구성요소 | 라이선스 | 출처 |
|---|---|---|
| OpenAI Whisper (코드·가중치) | MIT, Copyright (c) 2022 OpenAI | https://github.com/openai/whisper |
| Transformers.js (`@huggingface/transformers` 4.3.1) | Apache License 2.0 | https://github.com/huggingface/transformers.js |
| ONNX Runtime Web (Transformers.js가 불러옴) | MIT, Copyright (c) Microsoft Corporation | https://github.com/microsoft/onnxruntime |
| 변환 모델 `onnx-community/whisper-large-v3-turbo`, `whisper-small`, `whisper-base` | 저장소별 표기가 다름(2026-11-09 Hugging Face API 기준: turbo·small은 미표기, base는 apache-2.0) | https://huggingface.co/onnx-community |

- 변환 모델의 원본은 `openai/whisper-large-v3-turbo`, `openai/whisper-small`, `openai/whisper-base`이고, 원본 turbo 가중치는 MIT로 표기되어 있습니다(2026-11-09 확인). 변환본(ONNX·양자화)은 원본의 파생물이므로 원본의 MIT 조건(저작권 고지와 라이선스 문구 유지)을 따르는 것으로 보고 출처를 밝힙니다. 이는 법률 자문이 아닌 작성자의 판단입니다.
- 각 라이선스의 전문은 위 출처 링크의 `LICENSE` 파일을 참고하세요.
