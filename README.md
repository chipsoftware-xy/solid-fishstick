Stage 0 Baseline

Hardware
---------
CPU: Intel i5-11320H
RAM: 16 GB
GPU: Intel Iris Xe

Model
-----
Gemma 3 4B

Performance
-----------
Warm Load: 0.76 s
Generation: 2.02 s
Total: 3.51 s

Status
------
✓ Suitable for local voice assistant

saleperson/
│
├── app/
│   ├── llm/
│   │   └── ollama_client.py
│   │
│   ├── stt/
│   │   └── whisper.py
│   │
│   ├── tts/
│   │   └── piper.py
│   │
│   └── benchmark/
│       └── pipeline.py
│
├── models/
│
├── audio/
│   ├── input/
│   └── output/
│
├── requirements.txt
└── README.md

Gemma3:4b
