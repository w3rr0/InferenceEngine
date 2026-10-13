# InferenceEngine

An experimental hybrid inference engine for autoregressive language models.

The project combines:

- Python and PyTorch for model execution,
- Rust for request scheduling and memory management,
- PyO3 for communication between the Python and Rust components,
- paged KV-cache handling for batched text generation.

The current implementation focuses on Llama-style models and explores efficient prefill and decode execution.

## Status

This project is in an early stage of development. Its API, architecture, and supported features may change frequently.

It is currently intended for experimentation and development rather than production use.