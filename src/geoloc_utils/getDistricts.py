import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from jsonargparse import CLI

df_str = pd.read_csv(".src/datasets/dataset_streets.csv", index_col=0)

def extract_street_from_address(line: str) -> str:
    """
    Извлекает название улицы из адресной строки, очищая от лишних элементов.
    
    Аргументы:
        line (str): Адресная строка для обработки
        
    Возвращает:
        str: Очищенное название улицы или оригинальная строка при ошибке
    """

    try:
        street_str = line.split(",", maxsplit=1)[0].strip()
    except AttributeError:
        print(line)

    # Заменить подстроку на пустоту
    for pattern in [
        r"\u200b",
        r"ул\.",
        r"^пр\.",
        r"^пл\.",
        r"просп\.",
        r"проспект",
        r"Богдана",
        r"магистраль",
        r"ш\.",
        r"микрорайон",
        r"^пос\.",
    ]:
        street_str = re.sub(pattern, "", street_str)

    # Заменить подстркоу в конце на заданую подстроку:
    for pattern, new_substring in [(r"пл\.$", "площадь"), (r"o", "о")]:
        street_str = re.sub(pattern, new_substring, street_str)

    return street_str.strip()


def find_street_in_df(line: pd.Series, df: pd.DataFrame):
    """
    Находит улицу в справочнике улиц и определяет район.
    
    Аргументы:
        line (pd.Series): Строка DataFrame с информацией о месте
        df (pd.DataFrame): Справочник улиц с районами
        
    Возвращает:
        pd.Series: Исходная строка с добавленным полем 'district'
    """
    selection = df[df["Улица"].str.contains(line.street, case=False)]
    district = None
    if len(selection) > 0:
        district = selection["Район"].iloc[0]
    line["district"] = district
    return line


def update_district_in_desc(line: pd.Series):
    """
    Обновляет описание места, добавляя информацию о районе.
    
    Аргументы:
        line (pd.Series): Строка DataFrame с информацией о месте
        
    Возвращает:
        pd.Series: Обработанная строка без временных полей
    """
    if (line.district is not None) and ("Новосибирская" not in line.address):
        if "," not in line.district:
            line["description"] = line["description"].replace(
                "Новосибирск", f"{line.district} район, Новосибирск"
            )
        else:
            line["description"] = line["description"].replace(
                "Новосибирск", f"{line.response} район, Новосибирск"
            )
    return line.drop(
        ["district", "street", "response", "full_url", "request_url", "house"]
    )


def fetch_data(url):
    """
    Получает информацию о районе по URL.
    
    Аргументы:
        url (str): URL страницы с информацией об улице
        
    Возвращает:
        str: Название района или сообщение об ошибке
    """
    try:
        # print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        th_element = soup.find("th", text="Район")
        td_element = th_element.find_next("td")

        result = td_element.text.strip()
        result = result.replace(" район", "")
        return result
    except Exception as e:
        return f"Error: {e}"


def get_cafes_districts(df_cafes, output_path: Path):
    """
    Обрабатывает данные о кафе, добавляя информацию о районах.
    
    Аргументы:
        df_cafes (pd.DataFrame): DataFrame с информацией о кафе
        output_path (Path): Путь для сохранения результата
    """
    df_cafes["street"] = df_cafes.address.apply(extract_street_from_address)
    df_cafes = df_cafes.apply(lambda x: find_street_in_df(x, df_str), axis=1)
    # df_cafes = df_cafes[df_cafes['district'].str.contains(',', na=False)]
    multi_district_mask = df_cafes["district"].str.contains(",", na=False)
    df_cafes.loc[multi_district_mask, "house"] = df_cafes.loc[
        multi_district_mask, "address"
    ].apply(
        lambda x: re.search(r"\d+", x.split(",")[1]).group()
        if isinstance(x, str) and "," in x
        else None
    )

    street_to_url = dict(zip(df_str["Улица"], df_str["Ссылка"]))
    df_cafes["request_url"] = df_cafes["street"].map(street_to_url)

    # Формирование полного URL и запрос данных только для multi-district
    df_cafes.loc[multi_district_mask, "full_url"] = df_cafes.loc[
        multi_district_mask, "request_url"
    ] + df_cafes.loc[multi_district_mask, "house"].astype(str)
    df_cafes.loc[multi_district_mask, "response"] = df_cafes.loc[
        multi_district_mask, "full_url"
    ].apply(fetch_data)

    # Финальная обработка
    df_cafes = df_cafes.apply(update_district_in_desc, axis=1)
    df_cafes.to_csv(output_path)


def extract_house_number(address):
    """
    Извлекает номер дома из строки адреса, поддерживая различные форматы записи.
    
    Аргументы:
        address (str): Адресная строка для анализа
        
    Возвращает:
        Optional[str]: Номер дома или None, если не удалось извлечь
        
    Пример:
        >>> extract_house_number("ул. Ленина, 12")
        '12'
        >>> extract_house_number("дом 15/2 корпус А")
        '15/2'
    """

    if not isinstance(address, str) or not address.strip():
        return None

    try:
        # Основные паттерны для поиска номера дома
        patterns = [
            # Для формата "ул. Романова, 23 Новосибирск..." (ищем число после последней запятой перед текстом)
            r",\s*(\d+)\s+[А-Яа-яA-Za-z]",
            # Для формата "ул. Мастеров, 12" (число в конце после запятой)
            r",\s*(\d+)\s*$",
            # Для формата "ул.Свердлова, 13" (без пробела после запятой)
            r",\s*(\d+)",
            # Для формата "дом 5/1" (с дробями)
            r"(?:дом|д\.|строение|корпус|к\.|к)\s*(\d+[/-]?\d*[А-Яа-яA-Za-z]*)",
            # Для формата "строение 3А" (с буквами)
            r"(?:дом|д\.|строение|корпус|к\.|к)\s*(\d+[А-Яа-яA-Za-z]*)",
            # Резервный вариант - первое число в строке
            r"\b(\d+)\b",
        ]

        for pattern in patterns:
            if match := re.search(pattern, address, re.IGNORECASE):
                return match.group(1)

    except AttributeError:
        print(address)

    return None


def add_district_to_ents(line):
    """Определяет район для развлекательного заведения по его адресу.
    
    Аргументы:
        address (str): Адрес заведения для поиска
        
    Возвращает:
        Optional[str]: Название района или None, если не удалось определить
    """
    df_str = pd.read_csv("./src/datasets/dataset_streets.csv")
    for _, row in df_str.iterrows():
        street = row["Улица"]
        if (f"{street} " in line) or (f"{street}," in line):
            # print(_, row['Район'], row['Улица'])
            if "область" not in line or "обл" not in line or "район" not in line:
                return row["Район"]
    return None


def update_entert_districts(line):
    """
    Добавляет информацию о районе в описание развлекательного заведения.
    
    Аргументы:
        line (pd.Series): Строка данных о заведении
        
    Возвращает:
        pd.Series: Обработанная строка с обновленным описанием
    """

    if (line.district is not None) and ("Новосибирская" not in line.address):
        if "," not in line.district or "Error" in line.response:
            line["description"] += (
                f". {line['name']} располагается в районе: {line.district}, г. Новосибирск"
            )
            # line['description'] = line["description"].replace("Новосибирск", f"{line.district} район, Новосибирск")
        else:
            line["description"] += (
                f". {line['name']} располагается в районе: {line.response}, г. Новосибирск"
            )
        #  line['description'] = line["description"].replace("Новосибирск", f"{line.response} район, Новосибирск")
    return line.drop(
        ["district", "street", "response", "full_url", "request_url", "house"]
    )


def get_ents_districts(df_ent, output_path: Path):
    """
    Обрабатывает данные о развлекательных заведениях, добавляя информацию о районах.
    
    Аргументы:
        df_ent (pd.DataFrame): DataFrame с данными о заведениях
        output_path (Path): Путь для сохранения результата
    """

    df_ent["address"] = df_ent["address"].str.replace("м.", " м.")
    df_ent["description"] = df_ent["description"].str.replace("м.", " м.")
    df_ent["address"] = df_ent["address"].str.replace("ост.", " ост.")
    df_ent["description"] = df_ent["description"].str.replace("ост.", " ост.")
    df_ent["description"] = df_ent["description"].str.replace("Стоимость", " Стоимость")

    df_ent["street"] = df_ent.address.apply(extract_street_from_address)
    df_ent = df_ent.apply(lambda x: find_street_in_df(x, df_str), axis=1)

    mask = df_ent["district"].isna()  # или df_na['district'].isnull()
    df_ent.loc[mask, "district"] = df_ent.loc[mask, "address"].map(add_district_to_ents)

    multi_district_mask = df_ent["district"].str.contains(",", na=False)

    df_ent.loc[multi_district_mask, "house"] = df_ent["address"].apply(
        extract_house_number
    )
    # Создание URL
    street_to_url = dict(zip(df_str["Улица"], df_str["Ссылка"]))
    df_ent["request_url"] = df_ent["street"].map(street_to_url)

    # Формирование полного URL и запрос данных только для multi-district
    df_ent.loc[multi_district_mask, "full_url"] = df_ent.loc[
        multi_district_mask, "request_url"
    ] + df_ent.loc[multi_district_mask, "house"].astype(str)

    df_ent.loc[multi_district_mask, "response"] = df_ent.loc[
        multi_district_mask, "full_url"
    ].apply(fetch_data)

    df_ent = df_ent.apply(update_entert_districts, axis=1).to_csv(
        output_path
    )


def get_hotels_districts(df_hotels, output_path: Path):
    """Обрабатывает данные об отелях, добавляя информацию о районах.
    
    Args:
        df_hotels (pd.DataFrame): DataFrame с данными об отелях
        output_path (Path): Путь для сохранения результата    
    """

    df_hotels["street"] = df_hotels.address.apply(extract_street_from_address)
    df_hotels = df_hotels.apply(lambda x: find_street_in_df(x, df_str), axis=1)
    # df_cafes = df_cafes[df_cafes['district'].str.contains(',', na=False)]
    multi_district_mask = df_hotels["district"].str.contains(",", na=False)
    df_hotels.loc[multi_district_mask, "house"] = df_hotels.loc[
        multi_district_mask, "address"
    ].apply(extract_house_number)

    # Создание URL
    street_to_url = dict(zip(df_str["Улица"], df_str["Ссылка"]))
    df_hotels["request_url"] = df_hotels["street"].map(street_to_url)

    # Формирование полного URL и запрос данных только для multi-district
    df_hotels.loc[multi_district_mask, "full_url"] = df_hotels.loc[
        multi_district_mask, "request_url"
    ] + df_hotels.loc[multi_district_mask, "house"].astype(str)
    df_hotels.loc[multi_district_mask, "response"] = df_hotels.loc[
        multi_district_mask, "full_url"
    ].apply(fetch_data)

    # Финальная обработка
    df_hotels = df_hotels.apply(update_district_in_desc, axis=1)
    df_hotels.to_csv(output_path)


def main(
    *,
    hotels_csv: Path,
    ent_csv: Path,
    cafes_csv: Path,
):
    df_hotels = pd.read_csv(hotels_csv, index_col=0)
    df_hotels = df_hotels.drop("Unnamed: 0", axis=1)
    get_hotels_districts(
        df_hotels,
        output_path=hotels_csv.with_stem(
            f"{hotels_csv.stem}_districts",
        ),
    )

    df_ent = pd.read_csv(ent_csv, index_col=0)
    get_ents_districts(
        df_ent,
        output_path=ent_csv.with_stem(
            f"{ent_csv.stem}_districts",
        ),
    )

    df_cafes = pd.read_csv(cafes_csv, index_col=0)
    get_cafes_districts(
        df_cafes,
        output_path=cafes_csv.with_stem(
            f"{cafes_csv.stem}_districts",
        ),
    )

def cli():
    CLI(main, as_positional=False)
