# adapters

In v0.1 every agent is called through LiteLLM on Amazon Bedrock by `scripts/run_text.py`
(`--agent-llm bedrock/<model-id>`), so no per-system adapter is needed. The speech condition
(caller turns through Polly, a noisy 8 kHz phone line and Transcribe) lives in
`simulator/speech_channel.py`.

Adapters for full-duplex speech-to-speech agents (e.g. Amazon Nova 2.5 Sonic) and for other
cascades are planned for the next release; this folder will hold one adapter per such system.
