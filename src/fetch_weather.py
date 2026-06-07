"""Pobieranie danych pogodowych z OpenWeather API.

Klucz API nie jest wpisany w kodzie. Przed uruchomieniem ustaw zmienną środowiskową:
    OWM_API_KEY=twoj_klucz_openweather
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("OWM_API_KEY")

CITIES = [
    "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin", "Bydgoszcz",
    "Lublin", "Białystok", "Katowice", "Gdynia", "Częstochowa", "Radom", "Toruń",
    "Sosnowiec", "Rzeszów", "Kielce", "Gliwice", "Olsztyn", "Zabrze", "Bielsko-Biała",
    "Bytom", "Zielona Góra", "Rybnik", "Ruda Śląska", "Opole", "Tychy", "Gorzów Wielkopolski",
    "Elbląg", "Płock", "Dąbrowa Górnicza", "Wałbrzych", "Włocławek", "Tarnów", "Chorzów",
    "Koszalin", "Kalisz", "Legnica", "Grudziądz", "Jaworzno", "Słupsk", "Jastrzębie-Zdrój",
    "Nowy Sącz", "Jelenia Góra", "Siedlce", "Mysłowice", "Konin", "Piła", "Piotrków Trybunalski"
]


def fetch_weather_for_city(city: str) -> dict | None:
    """Pobiera aktualną pogodę dla jednego miasta."""
    if not API_KEY:
        raise RuntimeError(
            "Brak klucza API. Ustaw zmienną środowiskową OWM_API_KEY, zamiast wpisywać klucz w kodzie."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": f"{city},PL", "appid": API_KEY, "units": "metric", "lang": "pl"}

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if response.status_code != 200:
            print(f"✗ Błąd dla {city}: {data.get('message', 'Nieznany błąd')}")
            return None

        return {
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Miasto": city,
            "Temperatura": data["main"]["temp"],
            "Temp_Odczuwalna": data["main"]["feels_like"],
            "Cisnienie": data["main"]["pressure"],
            "Wilgotnosc": data["main"]["humidity"],
            "Predkosc_Wiatru": data["wind"]["speed"],
            "Zachmurzenie": data["clouds"]["all"],
            "Opis": data["weather"][0]["description"],
        }
    except requests.RequestException as exc:
        print(f"! Błąd połączenia dla {city}: {exc}")
        return None
    except KeyError as exc:
        print(f"! Nieoczekiwana struktura odpowiedzi API dla {city}. Brak pola: {exc}")
        return None


def fetch_weather_data(output_path: str = "data/pogoda_polska.csv") -> pd.DataFrame:
    """Pobiera dane pogodowe dla listy miast i zapisuje je do CSV."""
    weather_data: list[dict] = []
    print(f"Rozpoczynam pobieranie danych dla {len(CITIES)} miast...")

    for city in CITIES:
        row = fetch_weather_for_city(city)
        if row:
            weather_data.append(row)
            print(f"✓ Pobrano: {city}")
        time.sleep(0.1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(weather_data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 45)
    print(f"SUKCES! Dane zapisane w: {output_path}")
    print(f"Liczba poprawnie pobranych rekordów: {len(df)}")
    print("=" * 45)
    return df


if __name__ == "__main__":
    fetch_weather_data()
