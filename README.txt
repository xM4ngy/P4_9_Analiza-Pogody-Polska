Analiza Pogody Polska
=====================

Autorzy (Grupa 9):
- Majewski Aleksander – 73037
- Palade Mikołaj – 51204
- Ryczywolski Jakub – 73184
- Szatyłowicz Adam – 72327
- Uzar Maciej – 71594

Folder oddawany na Plato: P4_9_Analiza-Pogody-Polska_73184
Repozytorium projektu:
https://github.com/xM4ngy/P4_9_Analiza-Pogody-Polska
Wagi modelu (plt5-weather, plik duży – poza paczką):
https://drive.google.com/file/d/1FqQRWrjq-2u-QW-LCm4HdLodhEELa8RH/view?usp=drive_link

1. Opis projektu
----------------
Projekt dotyczy pobierania danych pogodowych dla wybranych miast w Polsce,
analizy korelacji pomiędzy parametrami pogodowymi oraz przygotowania części LLM
służącej do generowania krótkich raportów pogodowych w języku polskim.

Główna zaimplementowana funkcjonalność:
- pobieranie danych pogodowych z OpenWeather API,
- zapis danych do pliku CSV,
- analiza korelacji Pearsona,
- wygenerowanie heatmapy korelacji,
- przygotowanie datasetu wejście-wyjście dla modelu LLM,
- fine-tuning modelu allegro/plt5-small wraz z ewaluacją (BLEU/ROUGE).

2. Struktura katalogu
---------------------
P4_9_Analiza-Pogody-Polska_73184/
  README.txt
  main.py
  raport.pdf
  requirements.txt
  .env.example
  src/
    __init__.py
    fetch_weather.py
    analiza.py
    llm_finetuning.py
    notebook_analiza.ipynb
    notebook_finetuningNaColabie.ipynb
  data/
    pogoda_polska.csv
    llm_dataset.csv
  outputs/
    AnalizaZdjecie.png
    macierz_korelacji.csv
    llm_eval.json

3. Dane
-------
Dane są pobierane z OpenWeather API dla 50 miast w Polsce.
Do paczki dodano odtwarzalny plik data/pogoda_polska.csv, aby można było
uruchomić analizę bez klucza API. Ten plik ma 50 wierszy i jest jednym
snapshotem danych pogodowych z czasu: 2026-01-01T10:00:00+00:00.

Kolumny w data/pogoda_polska.csv:
- Timestamp_UTC,
- Miasto,
- Temperatura,
- Temp_Odczuwalna,
- Cisnienie,
- Wilgotnosc,
- Predkosc_Wiatru,
- Zachmurzenie,
- Opis.

Uwaga: pogoda jest zmienna w czasie. Po ponownym pobraniu danych z API wyniki
korelacji mogą różnić się od wyników z dołączonego pliku CSV.

4. Instalacja
-------------

Instalacja zależności:

pip install -r requirements.txt

5. Konfiguracja klucza API
--------------------------
Klucz OpenWeather API nie jest wpisany w kodzie. Należy ustawić zmienną
środowiskową OWM_API_KEY albo skorzystać z pliku .env utworzonego lokalnie na
podstawie .env.example.

Linux/macOS:
export OWM_API_KEY="twoj_klucz_openweather"

Windows PowerShell:
$env:OWM_API_KEY="twoj_klucz_openweather"

Do paczki nie należy dodawać prawdziwego pliku .env.

6. Uruchomienie kodu
--------------------
Analiza na dołączonych danych przykładowych:

python main.py --mode analyze

Wyniki:
- outputs/AnalizaZdjecie.png
- outputs/macierz_korelacji.csv

Pobranie aktualnych danych z API:

python main.py --mode fetch

Pobranie danych i wykonanie analizy w jednym kroku:

python main.py --mode all

Przygotowanie datasetu dla LLM:

python main.py --mode llm-dataset

Fine-tuning modelu LLM (zalecane GPU, np. Google Colab T4):

python src/llm_finetuning.py --train --epochs 50 --lr 3e-4

Wyniki treningu (metryki + przykłady generacji) zapisują się do
outputs/llm_eval.json.

7. Wyniki analizy korelacji dla dołączonego CSV
-----------------------------------------------
Wyniki zapisane w outputs/macierz_korelacji.csv i pokazane na heatmapie
outputs/AnalizaZdjecie.png są zgodne z raportem:
- Temp - Odczuw.: 0.90
- Wiatr - Odczuw.: -0.27
- Wilg. - Odczuw.: -0.43
- Zachm. - Temp: -0.04

Interpretacja:
- H1: w dołączonym CSV korelacja Wiatr - Odczuw. jest ujemna, ale słaba, więc hipoteza o silnej korelacji ujemnej nie została potwierdzona.
- H2: wilgotność ma silniejszy związek z temperaturą odczuwalną niż wiatr, więc hipoteza z etapu 2 nie została potwierdzona na tym snapshocie.
- H3: zachmurzenie i temperatura mają korelację bliską zeru, więc nie ma istotnej zależności liniowej.

8. Model LLM i fine-tuning
--------------------------
Jako model LLM wybrano allegro/plt5-small - Transformer typu Encoder-Decoder
dopasowany do zadania text-to-text (zamiana danych strukturalnych na opis w
języku naturalnym). Model generuje krótki raport pogodowy na podstawie danych:
miasto, temperatura, temperatura odczuwalna, ciśnienie, wilgotność, wiatr i
zachmurzenie.

Plik data/llm_dataset.csv zawiera 50 par input_text -> target_text utworzonych
na podstawie data/pogoda_polska.csv.

Fine-tuning przeprowadzono w Google Colab (GPU T4): 50 epok, learning rate 3e-4,
podział 40 train / 10 eval. Wyniki z outputs/llm_eval.json:
- eval_loss: 0.0161
- BLEU: 97.55
- ROUGE-1: 98.45, ROUGE-2: 98.41, ROUGE-L: 98.45

Uwaga: tak wysokie metryki wynikają w dużej mierze z silnie szablonowej
struktury danych (podstawianie liczb do stałego wzorca zdania), więc mierzą
raczej odtworzenie szablonu niż generalizację. Przy tak małym zbiorze możliwe
są halucynacje wartości liczbowych (przykład w raporcie). Wagi modelu
(outputs/plt5-weather/) są duże i udostępnione przez Google Drive (link wyżej).

9. Użycie AI
------------
AI zostało użyte w dwóch rolach:
- allegro/plt5-small jako element badawczy projektu do generowania opisów
  pogody na podstawie danych liczbowych,
- narzędzia AI (ChatGPT oraz Claude) pomocniczo: do uporządkowania raportu,
  README i struktury katalogu, usunięcia klucza API z kodu źródłowego,
  przygotowania kodu ewaluacji LLM (obliczanie BLEU/ROUGE) oraz doboru
  hiperparametrów fine-tuningu.
Wszystkie wartości liczbowe w raporcie pochodzą z faktycznie uruchomionego
kodu, a nie zostały wygenerowane przez AI.

10. Odtwarzanie wyników
-----------------------
Część statystyczna:

pip install -r requirements.txt
python main.py --mode analyze

Dataset LLM:

python main.py --mode llm-dataset

Fine-tuning:

python src/llm_finetuning.py --train --epochs 50 --lr 3e-4

W środowisku lokalnym trening może być wolny. Do pełnego eksperymentu zalecany
jest Google Colab z GPU i notatnik src/notebook_finetuningNaColabie.ipynb.
