# импорт необходимых библиотек
import contextlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from jsonargparse import CLI
from requests_futures.sessions import FuturesSession


# функция для сбора ссылок на категории развлечений
def get_categories_links(html_session, sight):
    request_content = html_session.get(sight, stream=True).result()
    request_content = request_content.content
    page_content=BeautifulSoup(request_content)

    main_section = page_content.find("div", class_="section-wrap")
    categories_links = [(block.find("span",class_="name").text, block['href']) for block in main_section.find_all('a')]
    return categories_links

# функция для сбора ссылок на развлечения
def get_entertainments_links(html_session, category):
    base_link = "https://welcome-novosibirsk.ru"
    request_content = html_session.get(f"{base_link}{category[1]}{'/?SHOWALL_1=1'}", stream=True).result()
    request_content = request_content.content
    page_content = BeautifulSoup(request_content, exclude_encodings=["maccyrillic","mac_greek","mac_latin2"])
    main_section = page_content.find("div", class_="section-wrap")
    try:
        entertainments_link = [[category[0], block.find("span",class_="name").text, block['href']] for block in main_section.find_all('a')]
    except AttributeError:
        print("error", f"{base_link}{category[1]}{'/?SHOWALL_1=1'}")
    return entertainments_link

# функция для удаления лишних переносов строки
def preprocess_description(description: str | None) -> str:
    if description is None:
        return None
    description_fields = description.strip("\n").split("\n\n\n")
    return [desc.replace("\n", " ") for desc in description_fields]

# функция для получения данных о развлечениях
def get_entertainments_description(html_session, place):
    base_link = "https://welcome-novosibirsk.ru"
    request_content = html_session.get(f"{base_link}{place[2]}", stream=True).result()
    request_content = request_content.content
    page_content = BeautifulSoup(request_content, exclude_encodings=["maccyrillic","mac_greek","mac_latin2"])
    info_fields = preprocess_description(page_content.find("div", class_="info-block").text)
    try:
        description_field = page_content.find("div", class_="tab-content").text
    except AttributeError:
        print("error no description", f"{base_link}{place[2]}")
    return place + [info_fields] + [description_field]

# основная функция для парсинга развлечений
def parse_entertainments():
    base_link = "https://welcome-novosibirsk.ru"
    sights_link = ["https://welcome-novosibirsk.ru/place-to-visit/sights/", "https://welcome-novosibirsk.ru/place-to-visit/active-leisure/", "https://welcome-novosibirsk.ru/place-to-visit/shopping/", "https://welcome-novosibirsk.ru/place-to-visit/beauty-health/"]

    session = FuturesSession(max_workers=10)

    # получение ссылок на категории развлечений
    futures = []
    categories_links = []
    with ThreadPoolExecutor(max_workers=10) as executor, contextlib.redirect_stderr(open(os.devnull, 'w')):
        for sight in sights_link:
            futures.append(executor.submit(get_categories_links, html_session=session, sight=sight))
        for categories in as_completed(futures):
            categories_links += categories.result()

    # получение ссылок на развлечения
    futures = []
    entertainments_link = []
    with ThreadPoolExecutor(max_workers=10) as executor, contextlib.redirect_stderr(open(os.devnull, 'w')):
        for categories in categories_links:
            futures.append(executor.submit(get_entertainments_links, html_session=session, category=categories))
        for entertainment in as_completed(futures):
            entertainments_link += entertainment.result()
    
    # получение данных о развлечениях
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor, contextlib.redirect_stderr(open(os.devnull, 'w')):
        for entertainment in entertainments_link:
            futures.append(executor.submit(get_entertainments_description, html_session=session, place=entertainment))
    parsed_places = []
    for future in as_completed(futures):
        parsed_places.append(future.result())

    return parsed_places

# функция для извлечения рабочего времени
def extract_work_time(info_list):
    for item in info_list:
        if 'Режим работы' in item:
            return item.replace("Режим работы ", "")
    return None  # если время не найдено

# функция для извлечения адреса
def extract_address(info_list):

    address = info_list[0]  # получаем первый элемент - он содержит адрес
    return address.replace('Адрес ', '')

# приведение адресов к виду улица-дом-город
def clean(line):
    pattern = re.compile(r"(?:(?:[^,]*,\s)?)(?P<street>(?:(?:ул|пр)\.\s[А-Я][^,]*)|(?:[А-Я].*)),\s(?P<house>\d+(?:/\d+)?(?:(?:[Аа])?(?=[^,]{0,3}\s))?)")

    '''
    (?:(?:[^,]*,\s)?) - пропуск необязательной части до улицы (например, индекс)

    улица
    (?:(?:ул|пр)\.\s[А-Я][^,]*)  - "ул." или "пр." с заглавной буквой
    или
    (?:[А-Я].*)  -  любая строка, начинающаяся с заглавной буквы

    дом
    \d+ - обязательная цифровая часть
    (?:/\d+)? - необязательный косой слэш с цифрами (для дробных домов)
    (?:(?:[Аа])? - необязательная буква "А" или "а" (корпус)
    (?=[^,]{0,3}\s))? - проверка того, это и правда дом (длина от запятой до запятой не больше 3 символов)
    '''
    match = pattern.search(line)
    
    try:
        clean_address = match.group(0)
    except AttributeError:
        clean_address = line
    
    if "Новосибирск" not in clean_address:
        clean_address = clean_address + ", Новосибирск"
    
    return clean_address

def main(output_path: Path):
    ent = parse_entertainments()

    entertainments = pd.DataFrame(ent)

    entertainments = entertainments.rename(columns={0: "category", 1: "name", 2: "uri", 3: "address", 4: "description"})

    # редактирование ссылки
    entertainments['uri'] = "https://welcome-novosibirsk.ru" + entertainments['uri']

    # очищение описания от управляющих последовательностей
    entertainments['description'] = entertainments['description'].str.replace(r'[\r\t]', '', regex=True).str.replace(r'[\n\xa0]',' ', regex=True)

    # создание столбца с рабочим временем и очищение адресов
    entertainments['work_time'] = entertainments['address'].apply(extract_work_time)
    entertainments['address'] = entertainments['address'].apply(extract_address)

    # убираем места без физического адреса
    entertainments = entertainments[entertainments['address'] != ""]
    entertainments = entertainments[~entertainments['address'].str.contains('Сайт')]
    entertainments = entertainments[~entertainments['address'].str.contains('Телефон')]

    # стандартизация адресов
    entertainments['address'] = entertainments['address'].apply(clean)

    # сохранение мест питания в файл
    pd.DataFrame(entertainments).to_csv(output_path) # "./src/datasets/dataset_entertainments.csv"
 
def cli():
    CLI(main)

if __name__ == "__main__":
    cli()