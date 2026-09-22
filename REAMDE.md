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

1. Odbiór i Tokenizacja (Rust - Serwer HTTP)
Serwer sieciowy (np. Axum) odbiera zapytanie REST lub WebSocket z tekstem. Używając biblioteki `tokenizers`, zamienia tekst promptu na tablicę liczb (`input_ids`). Utworzony obiekt `Request` trafia do kolejki `pending` w obiekcie Schedulera.

2. Alokacja Pamięci (Rust - Block Manager)
Przed dopuszczeniem żądania do obliczeń, Scheduler pyta Block Managera o dostępność pamięci. Jeśli prompt ma 35 tokenów, a jeden blok mieści 16 tokenów, Block Manager rezerwuje 3 fizyczne bloki w VRAM i przypisuje je do tego żądania, tworząc jego strukturę `block_tables` (mapowanie adresów logicznych na fizyczne).

3. Tworzenie Batcha (Rust - Scheduler)
W głównej pętli silnika Scheduler wybiera żądania z kolejki `pending` (wymagające fazy prefill) oraz `active` (wymagające wygenerowania kolejnego tokena). Grupuje ich tokeny i łączy ich indywidualne `block_tables` w jedną, zbiorczą strukturę. 

4. Przejście przez most FFI (Rust -> Python)
Za pomocą PyO3, z poziomu Rusta wywoływana jest funkcja Pythona. Rust przekazuje do niej zaledwie kilka argumentów: spłaszczony tensor wejściowy (tokeny), pozycje sekwencji dla każdego żądania oraz zbiorczą macierz `block_tables`.

5. Przejście przez Graf Modelu (Python - PyTorch)
Python przejmuje kontrolę. Przekazane dane przechodzą przez kolejne warstwy modelu LLM (np. Llama). Wykonywane są standardowe operacje embeddingu, warstwy MLP i normalizacje przy użyciu natywnych funkcji PyTorcha.

6. Wywołanie PagedAttention (Python -> Triton/CUDA)
Gdy sygnał dociera do warstwy uwagi (Attention), skrypt w Pythonie wywołuje dedykowany kernel PagedAttention. Kernel ten omija standardowe ciągłe bufory pamięci. Zamiast tego odczytuje dostarczoną macierz `block_tables` i pobiera klucze (Keys) oraz wartości (Values) bezpośrednio z pofragmentowanych bloków w pamięci fizycznej VRAM, generując wynik atencji.

7. Próbkowanie (Sampling) i Zwrot Tokenów (Python -> Rust)
Na samym końcu grafu wyliczane są prawdopodobieństwa dla kolejnych słów. Python dokonuje próbkowania (np. Argmax lub Top-K), uzyskując ID nowego tokena dla każdego zapytania w batchu. Tablica wygenerowanych tokenów jest zwracana przez granicę FFI z powrotem do Rusta.

8. Strumieniowanie do Klienta (Rust - Serwer HTTP)
Rust odbiera nowe ID tokenów, dekoduje je z powrotem na tekst i asynchronicznie przesyła do użytkownika otwartym kanałem (Server-Sent Events). Użytkownik widzi, jak na ekranie pojawia się kolejne słowo.

9. Aktualizacja Stanu i Cykl Życia (Rust - Scheduler)
Scheduler dopisuje wygenerowany token do historii żądania. Następnie weryfikuje zajętość pamięci: jeśli ostatni fizyczny blok przypisany do tego żądania został właśnie zapełniony nowym tokenem, Block Manager alokuje z puli VRAM jeden nowy, pusty blok i dodaje go do `block_tables`. Żądanie pozostaje w kolejce `active`. Całość wraca do kroku 3.

10. Zakończenie i Czyszczenie Pamięci (Rust - Block Manager)
Gdy zwrócony token to <EOS> (End of Sequence) lub osiągnięto maksymalną długość, serwer ostatecznie zamyka połączenie HTTP. Scheduler usuwa żądanie z kolejki, a Block Manager oznacza wszystkie jego fizyczne bloki w VRAM jako "wolne", natychmiast udostępniając tę pamięć dla nowych zapytań z kroku 2.
