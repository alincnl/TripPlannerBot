import math
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

def haversine(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние между двумя географическими точками по формуле Хаверсинуса.
    
    Параметры:
        lat1 (float): Широта первой точки в градусах
        lon1 (float): Долгота первой точки в градусах
        lat2 (float): Широта второй точки в градусах
        lon2 (float): Долгота второй точки в градусах
    
    Возвращает:
        float: Расстояние между точками в километрах
    """
    # радиус Земли в километрах
    R = 6371.0
    
    # переводим градусы в радианы
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # разница между координатами
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # формула Хаверсинуса
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # расстояние
    distance = R * c
    return distance

def get_coordinates_array(all_descr_df):
    """
    Находит координаты мест по их описаниям в объединенной базе данных.
    
    Параметры:
        all_descr_df (List): Список описаний мест для поиска
    
    Возвращает:
        List: Список кортежей вида (название, широта, долгота) для найденных мест. Если место не найдено, возвращает (None, None) для координат.
    """
    
    # загрузка данных
    all_places_df = pd.concat([
        pd.read_csv("./src/production_data/production_hotels_coord.csv", index_col=0),
        pd.read_csv("./src/production_data/production_cafes_coord.csv", index_col=0),
        pd.read_csv("./src/production_data/production_ent_coord.csv", index_col=0)
        #pd.read_csv("../production_data/production_hotels_coord.csv", index_col=0),
        #pd.read_csv("../production_data/production_cafes_coord.csv", index_col=0),
        #pd.read_csv("../production_data/production_ent_coord.csv", index_col=0)
    ], ignore_index=True)

    # столбец с описаниями места
    description_column = 'description'

    coordinates_list = []

    # перебор описаний, которые дал ретривер
    for descr in all_descr_df:
        # поиск строки, которая содержит нужное описание, в общей БД
        match = all_places_df[all_places_df[description_column].str.contains(descr[15:55], na=False, case=False, regex=False)]
        
        if not match.empty:
            coordinates_list.append((match['name'].values[0], match['latitude'].values[0], match['longitude'].values[0]))
        else:
            match = all_places_df[all_places_df[description_column].str.contains(descr[:15], na=False, case=False, regex=False)]
            if not match.empty:
                coordinates_list.append((match['name'].values[0], match['latitude'].values[0], match['longitude'].values[0]))
            else:
                match = all_places_df[all_places_df[description_column].str.contains(descr[:15], na=False, case=False)] 
                #print(f"Не смог найти место по описанию . Вот описание: {descr[:30]}")
                coordinates_list.append((None, None))
    return list(dict.fromkeys(coordinates_list))

def get_distances(coordinates_array):
    """
    Вычисляет расстояния между всеми парами мест и форматирует результат.
    
    Параметры:
        coordinates_array (List): Список кортежей с данными мест в виде (название, широта, долгота)
    
    Возвращает:
        str: Отформатированная строка с расстояниями между всеми парами мест
    """
    distance_matrix = []
    for i in range(len(coordinates_array)):
        for j in range(i+1, len(coordinates_array)):
            distance = haversine(coordinates_array[i][1], coordinates_array[i][2], coordinates_array[j][1], coordinates_array[j][2])
           # print(coordinates_array)
            distance_matrix.append(f"От {coordinates_array[i][0]} до {coordinates_array[j][0]} около {round(distance+1)} км")
    return '. '.join(distance_matrix)
 