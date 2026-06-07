"""Główny plik uruchomieniowy projektu.

Domyślnie uruchamia analizę korelacji na pliku data/pogoda_polska.csv,
żeby wyniki można było odtworzyć bez klucza API.

Przykłady:
    python main.py --mode analyze
    python main.py --mode fetch
    python main.py --mode all
    python main.py --mode llm-dataset
"""

from __future__ import annotations

import argparse

from src.analiza import analyze_weather
from src.fetch_weather import fetch_weather_data
from src.llm_finetuning import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza pogody w Polsce")
    parser.add_argument(
        "--mode",
        choices=["analyze", "fetch", "all", "llm-dataset"],
        default="analyze",
        help=(
            "analyze - analiza przykładowego CSV; fetch - pobranie danych z API; "
            "all - pobranie danych i analiza; llm-dataset - przygotowanie datasetu dla LLM"
        ),
    )
    parser.add_argument("--input", default="data/pogoda_polska.csv", help="Plik CSV z danymi pogodowymi")
    parser.add_argument("--output-image", default="outputs/AnalizaZdjecie.png", help="Ścieżka zapisu heatmapy")
    parser.add_argument("--output-corr", default="outputs/macierz_korelacji.csv", help="Ścieżka zapisu macierzy korelacji")
    args = parser.parse_args()

    if args.mode in {"fetch", "all"}:
        fetch_weather_data(output_path=args.input)

    if args.mode in {"analyze", "all"}:
        analyze_weather(input_path=args.input, output_image=args.output_image, output_matrix=args.output_corr)

    if args.mode == "llm-dataset":
        build_dataset(input_csv=args.input, output_csv="data/llm_dataset.csv")


if __name__ == "__main__":
    main()
