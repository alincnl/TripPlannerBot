from pathlib import Path

import pandas as pd
import requests
from jsonargparse import CLI
from tqdm.auto import tqdm

tqdm.pandas()
def get_yandex_coordinates(address):
    """
    Получает координаты (долготу и широту) для заданного адреса с помощью Yandex Geocoder API.
    
    Параметры:
        address (str): Адрес, для которого нужно получить координаты.
        
    Возвращает:
        pd.Series: Pandas Series с двумя значениями - долготой и широтой, 
        или None, если адрес не найден.

    Пример:
        >>> coords = get_yandex_coordinates("Москва, Красная площадь")
        >>> print(coords)
        0    37.617635
        1    55.755814
        dtype: float64
    """
    base_url = "https://geocode-maps.yandex.ru/1.x"
    response = requests.get(base_url, params={
        "geocode": address,
        "apikey": 'YOUR_API_KEY',
        "format": "json",
    })
    response.raise_for_status()
    found_places = response.json()['response']['GeoObjectCollection']['featureMember']

    if not found_places:
        print(address)
        return None

    most_relevant = found_places[0]
    lon, lat = most_relevant['GeoObject']['Point']['pos'].split(" ")
    return pd.Series([lon, lat])

def get_coordinates(places_paths: list[Path]):
    """
    Получает и сохраняет координаты мест из CSV-файлов с адресами.
    
    Для каждого места в переданных файлах определяет географические координаты
    (долготу и широту) с помощью Yandex Geocoder API и сохраняет результат
    в новые CSV-файлы с суффиксом '_coord'.

    Параметры:
        places_paths (list[Path]): Список путей к CSV-файлам с данными о местах.
                                  Каждый файл должен содержать столбец 'address'
                                  с адресами для геокодирования.

    Возвращает:
        None: Функция не возвращает значения, но сохраняет результаты в файлы.

    Пример:
        >>> get_coordinates([
        ...     Path("./data/hotels.csv"),
        ...     Path("./data/cafes.csv")
        ... ])
        Определяет координаты для всех адресов и сохраняет в hotels_coord.csv и cafes_coord.csv
    """
    # places_paths = ["./src/production_data/production_hotels.csv", "./src/production_data/production_cafes.csv", "./src/production_data/production_ent.csv"]
    
    # определяем координаты и сохраняем в csv
    for place_path in places_paths:
        place_path = Path(place_path)
        place_df = pd.read_csv(place_path, index_col=0)

        print(f"Определяем координаты для мест из файла {place_path.stem}")
        place_df[['longtitude', 'latitude']] = place_df["address"].progress_apply(get_yandex_coordinates)
        place_df.to_csv(f"./src/production_data/{place_path.stem}_coord.csv")
 
def cli():
    CLI(get_coordinates)

if __name__ == "__main__":
    cli()