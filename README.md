# Inference Engine

## Overview

|component| language/technology | role and justification |
|---------|---------------------|------------------------|
|servew api and streaming| None                | async IO                |
|scheduler (continous batching)|None|gluing queries into one|
|Block Manager|Rust|GPU memory management|
|FFI (Python-Rust)|PyO3|Make rust code callable from Python|
|Computional graph and weights|Python - PyTorch|Initialize model and weights, load into VRAM, build neural network|
|Kernels|CUDA via Python (Triton)|Kernels for managing memory|

## Workflow

1. INICJALIZACJA (Python & Rust)
   Python wczytuje wagi modelu i inicjalizuje obiekt Engine z modułu napisanego w Ruście (np. `engine = Engine(config)`). Engine wewnętrznie tworzy Schedulera, Block Managera i (docelowo) odpala serwer API w tle, który zaczyna przyjmować żądania od klientów.

2. ŻĄDANIE NOWEJ PRACY (Python -> Rust)
   Główna pętla w Pythonie wywołuje `batch = engine.get_next()`. Jeśli nie ma nowych ani trwających requestów, Rust zwalnia blokadę GIL (Python::allow_threads) i usypia ten wątek, dopóki w tle nie pojawi się nowe żądanie.

3. BUDOWANIE BATCHA (Rust)
   Gdy są dostępne requesty, Scheduler wybiera odpowiednią ich liczbę z kolejek `pending` (nowe prompt'y) i `active` (w trakcie generacji). Block Manager alokuje wirtualne i fizyczne bloki w KV Cache dla tych żądań i generuje mapowanie pamięci.

4. ZWROT METADANYCH (Rust -> Python)
   Funkcja `get_next()` zwraca do Pythona obiekt `BatchData`. Zawiera on wyłącznie płaskie informacje niezbędne dla GPU: wektor wejściowych `input_ids`, złączoną macierz `block_tables` oraz długości kontekstów (sequence lengths).

5. FORWARD PASS I PAGED ATTENTION (Python - PyTorch/GPU)
   Python przekazuje metadane do warstw modelu (Llama). Obliczenia wykonywane są na GPU. Gdy dane docierają do warstwy Attention, wykorzystywane są przekazane `block_tables` do zlokalizowania i pobrania pofragmentowanych bloków KV Cache bezpośrednio z fizycznej pamięci VRAM.

6. PRÓBKOWANIE - SAMPLING (Python)
   Z wygenerowanych w ostatniej warstwie logitów wyliczane są ID nowych tokenów (np. przez argmax dla każdego zapytania w batchu).

7. PRZEKAZANIE WYNIKÓW (Python -> Rust)
   Python wywołuje `engine.step(new_tokens)`, przekazując listę nowo wygenerowanych ID z powrotem do Rusta.

8. AKTUALIZACJA STANU (Rust)
   Rust odbiera nowe tokeny. Scheduler przypisuje je do odpowiednich requestów.
- Jeżeli token to <EOS> (Koniec), request jest kończony, Block Manager zwalnia przypisaną mu pamięć, a (docelowo) klient HTTP dostaje sygnał o zakończeniu.
- Jeżeli request wymaga dalszej generacji, Block Manager sprawdza, czy ostatni blok jest pełny - jeśli tak, alokuje nowy fizyczny blok KV Cache, a request wraca do kolejki `active`.

9. ZAPĘTLENIE
   Proces wraca do punktu 2 - Python od razu ponownie wywołuje `engine.get_next()`.

## Project Structure

```text
my_infer_engine/
├── Cargo.toml               # Zależności Rust (pyo3, tokio, axum, serde, itp.)
├── pyproject.toml           # Konfiguracja Maturin (zależności Pythona, np. torch)
├── .gitignore
├── README.md
│
├── src/                     # CZĘŚĆ RUSTOWA (Control Plane)
│   ├── lib.rs               # Punkt wejścia FFI (PyO3). Eksportuje tylko klasę Engine.
│   ├── engine.rs            # Rdzeń Rusta: łączy Schedulera, Block Managera i komunikację z Tokio.
│   │
│   ├── memory/              # MODULE: ZARZĄDZANIE PAMIĘCIĄ (VRAM)
│   │   ├── mod.rs           # Udostępnia publiczne interfejsy modułu.
│   │   ├── allocator.rs     # Główna logika (alokacja/zwalnianie fizycznych bloków, OOM).
│   │   ├── block.rs         # Definicje struktur (Block, BlockTable).
│   │   └── tests.rs         # Testy jednostkowe Rusta dla Block Managera (krytyczne!).
│   │
│   ├── scheduler/           # MODULE: BATCHING I KOLEJKOWANIE
│   │   ├── mod.rs
│   │   ├── batch.rs         # Logika sklejania zapytań (prefill i decode) w jeden Batch.
│   │   └── request.rs       # Cykl życia i stan pojedynczego zapytania.
│   │
│   ├── server/              # MODULE: WARSTWA SIECIOWA I ASYNCHRONICZNOŚĆ
│   │   ├── mod.rs
│   │   ├── router.rs        # Konfiguracja Axum (endpointy REST).
│   │   └── sse.rs           # Implementacja Server-Sent Events (strumieniowanie tokenów).
│   │
│   └── api/                 # MODULE: MODELE DANYCH (Komunikacja ze światem)
│       ├── mod.rs
│       └── openai.rs        # Struktury żądań/odpowiedzi kompatybilne z OpenAI API.
│
├── my_infer_engine/         # CZĘŚĆ PYTHONOWA (Data Plane)
│   ├── __init__.py          # Ładuje skompilowaną binarkę Rusta (`from .my_infer_engine import *`).
│   ├── __main__.py          # Punkt startowy aplikacji (pozwala uruchomić: `python -m my_infer_engine`).
│   ├── runner.py            # Główna pętla generacji: wywołuje `get_next()`, model i `step()`.
│   │
│   ├── models/              # MODULE: ARCHITEKTURY SIECI (PyTorch)
│   │   ├── __init__.py
│   │   ├── llama.py         # Implementacja warstw (MLP, Norm) dla np. TinyLlama.
│   │   └── weights.py       # Logika ładowania plików `.safetensors`.
│   │
│   └── attention/           # MODULE: MECHANIZMY UWAGI
│       ├── __init__.py
│       ├── flex_attn.py     # Implementacja PagedAttention przy użyciu PyTorch (na początek).
│       └── triton_kernel.py # Przyszłościowa, szybka implementacja w OpenAI Triton.
│
└── tests/                   # TESTY (Python / Integracyjne)
├── test_attention.py    # Sprawdza czy Twoje PagedAttention zwraca to samo co naiwna atencja.
└── test_e2e.py          # Zrzuca testowe zapytania przez HTTP i sprawdza wyniki.
```