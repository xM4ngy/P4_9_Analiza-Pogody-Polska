"""Analiza korelacji Pearsona dla danych pogodowych."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLUMNS = [
    "Temperatura",
    "Temp_Odczuwalna",
    "Cisnienie",
    "Wilgotnosc",
    "Predkosc_Wiatru",
    "Zachmurzenie",
]
SHORT_NAMES = ["Temp", "Odczuw.", "Ciśn.", "Wilg.", "Wiatr", "Zachm."]


def analyze_weather(
    input_path: str = "data/pogoda_polska.csv",
    output_image: str = "outputs/AnalizaZdjecie.png",
    output_matrix: str = "outputs/macierz_korelacji.csv",
) -> pd.DataFrame:
    """Liczy korelację Pearsona i zapisuje heatmapę oraz CSV z macierzą korelacji."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku {input_path}. Najpierw pobierz dane albo użyj pliku przykładowego.")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn wymaganych do analizy: {missing}")

    df_small = df[COLUMNS].copy()
    df_small.columns = SHORT_NAMES
    corr = df_small.corr(method="pearson")

    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    Path(output_matrix).parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(output_matrix, encoding="utf-8-sig")

    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 12})
    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Analiza korelacji - Projekt Pogoda", pad=20, fontsize=15)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"--- GOTOWE! Heatmapa: {output_image}; macierz korelacji: {output_matrix} ---")
    return corr


if __name__ == "__main__":
    analyze_weather()
