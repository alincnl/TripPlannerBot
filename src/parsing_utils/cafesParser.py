# импорт необходимых библиотек
import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from jsonargparse import CLI


# функция получения количества страниц на сайте
def get_numpages(base_link, city):
    request_content = requests.get(f"{base_link}?page=1&page_size=24&city={city}").text
    json_content = json.loads(request_content)
    return json_content['num_pages']

# функция получения данных о местах питания на странице
def get_cafes(html_session, base_link, page_number, city):
    request_content = html_session.get(f"{base_link}?page={page_number}&page_size=24&city={city}").text
    json_content = json.loads(request_content)
    return json_content['results']

#9
#page=1&page_size=24&city=2586 

# функция парсинга мест питания со всех страниц сайта
def parse_cafes():
    base_link = "https://www.afisha.ru/rests/api/public/v1/restaurant/"
    cities = [9] #2586 - Бердск
    parsed_places = []
    for i in range(len(cities)):
        pages_number = get_numpages(base_link, city = cities[0])
        session = requests.Session()
        
        # разделение парсинга на 10 потоков
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor, contextlib.redirect_stderr(open(os.devnull, 'w')):
            for g in range(1, pages_number+1):
                futures.append(executor.submit(get_cafes, html_session=session, base_link = base_link, page_number=g, city = cities[0]))
        for future in as_completed(futures):
            parsed_places+=future.result()

    return parsed_places

# извлечение времени работы и среднего чека
def extract_time_n_bill(row):
    work_time = row['work_time']
    average_bill = row['average_bill']['name']
    return pd.Series([work_time, average_bill])

# извлечение направленностец кухни
def extract_cuisine(row):
    try:
        cuisines = [cuisine['name'] for cuisine in row['cuisine']]
    except TypeError:
        cuisines = ["Неизвестно"]
    return ', '.join(cuisines) 

def main(output_path: Path):
    # запуск сбора данных
    cafes = parse_cafes()

    cafes = pd.DataFrame(cafes)

    # оставляем только нужные строки
    cafes = cafes[['name', 'uri','address','description','rating','latitude','longitude','restaurant_type','tags','extra_info']]

    # добавление столбцов "Режим работы", "Средний чек" и "Кухни"
    cafes[['work_time', 'average_bill']] = cafes['extra_info'].apply(extract_time_n_bill)
    cafes['cuisine'] = cafes['tags'].apply(extract_cuisine)

    # удаление исходных столбцов
    cafes.drop(columns=['extra_info'], inplace=True)
    cafes.drop(columns=['tags'], inplace=True)

    # сохранение мест питания в файл
    pd.DataFrame(cafes).to_csv(output_path) # "./src/datasets/dataset_cafes.csv"

def cli():
    CLI(main)

if __name__ == "__main__":
    cli()