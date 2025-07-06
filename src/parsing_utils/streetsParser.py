import pandas as pd
import requests
from bs4 import BeautifulSoup


def parse_streets():
    """
    Парсит список улиц Новосибирска с их районами и ссылками с сайта ginfo.ru.
    
    Собирает данные по всем административным районам города, объединяя улицы,
    которые могут относиться к нескольким районам.

    Возвращает:
        pd.DataFrame: DataFrame с колонками:
            - 'Район': название района(ов)
            - 'Улица': название улицы
            - 'Ссылка': URL страницы с информацией об улице

    Пример:
        >>> streets_df = parse_streets()
        >>> print(streets_df.head())
               Район           Улица                              Ссылка
        0  Центральный  ул. Ленина  https://nsk.ginfo.ru/ulicy/ulica_lenina/
        1  Центральный  ул. Кирова  https://nsk.ginfo.ru/ulicy/ulica_kirova/
    """

    urls = []
    streets = []
    districts = []
    main_url = 'https://nsk.ginfo.ru'
    district_urls = ['https://nsk.ginfo.ru/ulicy/?rayon=447', 'https://nsk.ginfo.ru/ulicy/?rayon=448', 'https://nsk.ginfo.ru/ulicy/?rayon=449','https://nsk.ginfo.ru/ulicy/?rayon=450','https://nsk.ginfo.ru/ulicy/?rayon=451','https://nsk.ginfo.ru/ulicy/?rayon=452','https://nsk.ginfo.ru/ulicy/?rayon=453','https://nsk.ginfo.ru/ulicy/?rayon=454','https://nsk.ginfo.ru/ulicy/?rayon=455','https://nsk.ginfo.ru/ulicy/?rayon=456']

    for district_url in district_urls:
        response = requests.get(district_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        rayon_name = soup.find('div', modal='modal_rayon').find('span').text
        rayon_streets = soup.find_all('a', class_='ulica_link')

        for street in rayon_streets:
            street_name = street.span.next_sibling
            if street_name in streets:
                index = streets.index(street_name)
                if districts[index] != rayon_name:
                    districts[index] += f', {rayon_name}'
                    # print(districts[index], streets[index])
            else:
                #print(street['href'])
                urls.append(main_url+street['href'])
                streets.append(street_name)
                districts.append(rayon_name)
    return pd.DataFrame(data = {'Район': districts, 'Улица':streets, 'Ссылка':urls})

def main():
    streets = parse_streets()
    pd.DataFrame(streets).to_csv("./src/datasets/dataset_streets.csv")
 
if __name__ == "__main__":
    main()