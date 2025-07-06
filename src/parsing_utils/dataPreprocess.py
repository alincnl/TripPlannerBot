import re
from pathlib import Path

import pandas as pd
from jsonargparse import CLI


def compile_cafes_metadata(row: pd.Series) -> pd.Series:
    """
    Превращает данные о кафе в цельный текст с его описанием

    Аргументы:
        row (pd.Series): Строка данных, содержащая информацию о кафе

    Возвращает:
        pd.Series: Серия с ключами 'name', 'address', 'description'
    """

    if ("Новосибирск" or 'Бердск') not in row.address:
        row.address = row.address + ", Новосибирск"

    return pd.Series({
        'name': row.place_name,
        'address': row.address,
        'description':
        f"{row.restaurant_type} с названием '{row.place_name}' "
        f"по адресу {row.address} с рейтингом {row.rating} и средним чеком {row.average_bill}. '{row.place_name}' специализируется на {row.cuisine}"
        f". График работы: {row.work_time}. Сайт: {row.uri}" 
    })

def compile_hotels_metadata(row: pd.Series) -> pd.Series:
    """
    Превращает данные об отеле в цельный текст с его описанием

    Аргументы:
        row (pd.Series): Строка данных, содержащая информацию об отеле

    Возвращает:
        pd.Series: Серия с ключами 'name', 'address', 'description'
    """

    return pd.Series({
        'name': row.place_name,
        'address': row.address,
        'description':
        f"{row.place_name} по адресу {row.address}. Минимальная стоимость в отеле  '{row.place_name}' за ночь: {row.price_per_night}"
    })

def compile_entertainments_metadata(row: pd.Series) -> pd.Series:
    """
    Превращает данные о развлечении в цельный текст с его описанием

    Аргументы:
        row (pd.Series): Строка данных, содержащая информацию о развлечении

    Возвращает:
        pd.Series: Серия с ключами 'name', 'address', 'description'
    """
    return pd.Series({
        'name': row.place_name,
        'address': row.address,
        'description': f"Развлечение с названием '{row.place_name}'. {row.description}. График работы '{row.place_name}': {row.work_time}. Категория отдыха '{row.place_name}': {row.category}"
    })

def clean_text(text):
    """
    Очищает текст от лишних пробелов, невидимых символов и заменяет пустые значения

    Аргументы:
        text (str): Исходный текст для очистки

    Возвращает:
        str: Очищенный текст
    """
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\ufeff', '')  
        text = text.replace('nan', 'необходимо уточнить') 
        text = text.replace('с рейтингом 0.0 и', 'со')
        text = re.sub(r'(: [0-9+]\n0:)', ' ', text).strip()
    return text

def compile_production_dataset(
        *,
        hotels_csv: Path,
        nsk_cafes_csv: Path,
        nso_cafes_csv: Path,
        ent_csv: Path,
        nso_nature_csv: Path,
        output_folder: Path,
):
    """
    Подготовка спарсенных датасетов к использованию в RAG-пайплане.
    """
    
    hotels_df = pd.read_csv(hotels_csv) # './src/datasets/dataset_hotels.csv'
    nsk_cafes_df = pd.read_csv(nsk_cafes_csv) # './src/datasets/dataset_cafes.csv'
    nso_cafes_df = pd.read_csv(nso_cafes_csv,  index_col=0) # './src/datasets/compiled_cafes_nso.csv'
    entertainments_df = pd.read_csv(ent_csv) # './src/datasets/dataset_entertainments.csv'
    nsonature_df = pd.read_csv(nso_nature_csv, index_col=0, on_bad_lines="warn") # './src/datasets/dataset_nature.csv'

    hotels_df = hotels_df[~hotels_df['address'].str.contains(r'[a-zA-Z]', na=False, regex=True)]

    hotels_df.rename({"name": "place_name"}, inplace=True, axis=1)
    nsk_cafes_df.rename({"name": "place_name"}, inplace=True, axis=1)
    entertainments_df.rename({"name": "place_name"}, inplace=True, axis=1)
    nsonature_df.rename({"name": "place_name"}, inplace=True, axis=1)

    compiled_hotels_series = hotels_df.apply(compile_hotels_metadata, axis=1).map(clean_text)
    compiled_cafes_series = nsk_cafes_df.dropna(subset="address")
    compiled_cafes_series = compiled_cafes_series.apply(compile_cafes_metadata, axis=1).map(clean_text)
    compiled_entertainments_series = entertainments_df.apply(compile_entertainments_metadata, axis=1).map(clean_text)
    compiled_nsonature_series = nsonature_df.apply(compile_entertainments_metadata, axis=1).map(clean_text)

    compiled_all_ent_series = pd.concat([compiled_entertainments_series,compiled_nsonature_series], ignore_index=True)
    compiled_all_cafes_series = pd.concat([compiled_cafes_series,nso_cafes_df], ignore_index=True)

    output_folder.mkdir()
    compiled_hotels_series.to_csv(output_folder / "production_hotels.csv") # "./src/production_data/production_hotels.csv"
    compiled_all_cafes_series.to_csv(output_folder / "production_cafes.csv") # "./src/production_data/production_cafes.csv"
    compiled_all_ent_series.to_csv(output_folder / "production_ent.csv") # "./src/production_data/production_ent.csv"
 
def cli():
    CLI(compile_production_dataset, as_positional=False)

if __name__ == "__main__":
    cli()