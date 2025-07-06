# импорт необходимых библиотек
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from jsonargparse import CLI


# определение базового класса для представления объекта
@dataclass
class Place:
    name: str
    address: str
    closed = False

    # преобразование объекта класса в словарь
    def asdict(self) -> dict[str, Any]:
        class_dict = {"type": self.__class__.__name__}
        for field_name in self.__dataclass_fields__.keys():
            class_dict.update({field_name: getattr(self, field_name)})
        return class_dict

# определение класса для отелей
@dataclass
class Hotel(Place):
    price_per_night: int

# определение структуры для поиска данных
class SearchMetadata(NamedTuple):
    html_tag_name: str
    html_tag_data: str
    target_attr: str | None

# функция для парсинга веб-страницы
def parse_webpage(page_content : BeautifulSoup, target_class, search_criterias: list[SearchMetadata], base_search_criteria = None):
    parse_results : list[target_class] = []

    base_blocks = page_content.find_all(base_search_criteria.html_tag_name, base_search_criteria.html_tag_data)
    for html_block in base_blocks:
        parsed_params = {}
        none_param = False
        for search_metadata in search_criterias:
            param = html_block.find(search_metadata.html_tag_name, search_metadata.html_tag_data)
            
            if param is not None:
                param = param.text.strip()
                parsed_params[search_metadata.target_attr] = param
            else:
                none_param = True

        if not none_param:
            parse_results.append(target_class(**parsed_params))


    return parse_results

# функция для получения страницы и ее парсинга
def get_hotels_by_page(html_session: requests.Session, pageid : int):
    baselink = "https://ostrovok.ru/hotel/russia/p/western_siberia_novosibirsk_oblast_multi/?page="

    base_search = SearchMetadata(html_tag_name="div", html_tag_data='zen-hotelcard-dateless', target_attr=None)

    search_metadata = [
        SearchMetadata(html_tag_name="a", html_tag_data='zen-hotelcard-name-link', target_attr="name"),
        SearchMetadata(html_tag_name="p", html_tag_data='zen-hotelcard-address', target_attr="address"),
        SearchMetadata(html_tag_name="div", html_tag_data='zen-hotelcard-rate-price-value', target_attr="price_per_night"),
    ]

    request_content = html_session.get(f"{baselink}{pageid}", stream=True)
    if "Not Found" not in request_content.text:
        request_content = request_content.content
        page_content=BeautifulSoup(request_content)
        parse_results = parse_webpage(page_content, Hotel, search_metadata, base_search)
        return parse_results
    else:
        print(f"No hotels in page {pageid}")
        return []


def main(output_path: Path):
    # создание сессии и определение переменных
    session = requests.Session()
    pages = list(range(1,82))
    hotels = []
    nso_hotels = []

    # разделение парсинга отелей на 10 потоков
    with ThreadPoolExecutor(max_workers=10) as executor:
        for page in pages:
            hotels.append(executor.submit(get_hotels_by_page, html_session=session, pageid=page))
        for hotel in as_completed(hotels):
            nso_hotels += hotel.result()


    # удаление апартаментов и квартир
    nso_hotels = [hotel for hotel in nso_hotels if 'Апартаменты' not in hotel.name]
    nso_hotels = [hotel for hotel in nso_hotels if 'Apartments' not in hotel.name]
    nso_hotels = [hotel for hotel in nso_hotels if 'комнаты' not in hotel.name]
    nso_hotels = [hotel for hotel in nso_hotels if 'Квартира' not in hotel.name]

    # очищение цены
    for hotel in nso_hotels:
        hotel.price_per_night = hotel.price_per_night.replace('\n', " ")

    # сохранение отелей в файл
    pd.DataFrame(nso_hotels).to_csv(output_path) # "./src/datasets/dataset_hotels.csv"
 
def cli():
    CLI(main)

if __name__ == "__main__":
    cli()