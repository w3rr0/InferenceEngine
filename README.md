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