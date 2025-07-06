import json
import math
import re

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from tqdm import tqdm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# функция для создания ретривера
def make_retriver(name: str, documents: list, embedding_generator, text_splitter, search_type: str = "similarity", search_kwargs: dict = {"k": 7}):
    text_splits = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(collection_name=name, documents=text_splits, embedding=embedding_generator)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def docs_cleaning(docs):
    for i in range (len(docs)):
        cleaned_content = re.sub(r'(: [0-9+]\n0:)', ' ', docs[i].page_content).strip()
        docs[i].page_content = cleaned_content
        docs[i].page_content = docs[i].page_content.replace("description: ", "")
    return docs

# функция для определения координат места по его адресу
def get_yandex_coordinates(address):
    base_url = "https://geocode-maps.yandex.ru/1.x"
    response = requests.get(base_url, params={
        "geocode": address,
        "apikey": apikey,
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

# функция для подсчета расстояния между двумя точками на поврехности Земли
def haversine(lat1, lon1, lat2, lon2):
    # радиус Земли в километрах
    R = 6371.0
    
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # формула Хаверсинуса
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

# функция, которая сопоставляет найденные ретривером места с общей базой данных и возвращает их координаты
def get_coordinates_array(all_descr_df):
    all_places_df = pd.concat([
        pd.read_csv("../datasets/compiled_toy_hotels_coord.csv", index_col=0),
        pd.read_csv("../datasets/compiled_toy_cafes_coord.csv", index_col=0),
        pd.read_csv("../datasets/compiled_toy_entertainments_coord.csv", index_col=0)
    ], ignore_index=True)

    description_column = 'description'

    coordinates_list = []

    # перебор описаний, которые дал ретривер
    for descr in all_descr_df:
        # поиск строки, которая содержит нужное описание, в общей БД
        match = all_places_df[all_places_df[description_column].str.contains(descr[15:55], na=False, case=False)]
        
        if not match.empty:
            coordinates_list.append((match['name'].values[0], match['latitude'].values[0], match['longitude'].values[0]))
        else:
            match = all_places_df[all_places_df[description_column].str.contains(descr[:15], na=False, case=False)] 
            if not match.empty:
                coordinates_list.append((match['name'].values[0], match['latitude'].values[0], match['longitude'].values[0]))
            else:
                match = all_places_df[all_places_df[description_column].str.contains(descr[:15], na=False, case=False)] 
                print(f"Не смог найти место по описанию . Вот описание: {descr[:30]}")
                coordinates_list.append((None, None))

    return list(dict.fromkeys(coordinates_list))

# функция, которая по данноиу массиву с координатами возвращает строку с информацией о расстоянии между соответствующими местами
def get_distances(coordinates_array):
    distance_matrix = []
    for i in range(len(coordinates_array)):
        for j in range(i+1, len(coordinates_array)):
            distance = haversine(coordinates_array[i][1], coordinates_array[i][2], coordinates_array[j][1], coordinates_array[j][2])
            distance_matrix.append(f"От {coordinates_array[i][0]} до {coordinates_array[j][0]} около {round(distance+1)} км")
    return '. '.join(distance_matrix)

# функция для определения необходимых для вопроса на ответ категорий развлечений
def get_retriver_categories(llm, question):
    template = """Есть вопрос от пользователя: {question} 
    # Оцени, будет ли пользователю полезна для ответа на вопрос информация о: местах проживания, местах питания и развлечениях. Чтобы определить надобность, поставь себя на место этого пользователя, не додумывай вопрос пользователя, отвечай четко по тому что спрашивается. Если же просят составить маршрут более чем на 1 день, то отвечай [true, true, true].
    В ответе напиши  список из true и false, где на первом месте будет надобность мест проживания, на втором - мест питания (рестораны, кафе и прочее) и на третьем - развлечения (бассейны, парки и прочее). К примеру, [false, false, true].
    """
    prompt = ChatPromptTemplate.from_template(template)
    result = llm.invoke(prompt.format(question=question))

    try:
        result = result.content
    except AttributeError:
        result = result

    pattern = re.compile(r"\[.*\]")
    match = pattern.search(result)
    print(match)
    try:
        return json.loads(match[0])
    except:
        return [True,True,True]

# класс для хранения контекста
class RawContext:
    def __init__(self, context_name: str = "Unknown", 
                 retriever: VectorStoreRetriever = None, 
                 context_default_output: str = "нет необходимости"):
        self.context_name = context_name
        self.retriever = retriever
        self.context_default_output = context_default_output

# функция для получения необходимого контекста
def get_contexts_(llm, question: str, raw_contexts: list[RawContext]):
    categories_bool_list = get_retriver_categories(llm, question)
    
    result_context = {
        "question": question,
    }
    for raw_context, is_needed in zip(raw_contexts, categories_bool_list):
        if is_needed:
            extracted_context = (raw_context.retriever | format_docs).invoke(question)
        else:
            extracted_context = raw_context.context_default_output
        result_context[raw_context.context_name] = extracted_context
    
    return result_context

# функция для ответа на вопрос
def run_rag_chain(llm, question, prompt, prompt_paths,hotels_retriever, cafes_retriever, ent_retriever):
    contexts = get_contexts_(llm, question, raw_contexts=
                             [
                                RawContext(context_name="context_hotels", retriever=hotels_retriever),
                                RawContext(context_name="context_cafes", retriever=cafes_retriever),
                                RawContext(context_name="context_ent", retriever=ent_retriever)
                             ])

    descr_cafes = []
    descr_hotels = []
    descr_ent = []

    retr_result = contexts

    empty_cat = 0

    for retr in retr_result:
        if retr == 'context_hotels' and retr_result[retr] != 'нет необходимости':
            descr_hotels += retr_result[retr].split(sep='\n\n')
        if retr == 'context_ent' and retr_result[retr] != 'нет необходимости':
            descr_ent += retr_result[retr].split(sep='\n\n')
        if retr == 'context_cafes' and retr_result[retr] != 'нет необходимости':
            descr_cafes += retr_result[retr].split(sep='\n\n')
        if retr_result[retr] == 'нет необходимости':
            empty_cat += 1

    all_descr_df = descr_cafes + descr_hotels + descr_ent

    final_prompt = prompt.format(**contexts) 
    response = llm.invoke(final_prompt)  

    if ("маршрут" not in response.content.lower()) and ("план" not in response.content.lower()):
        return response.content
    
    coord_array = get_coordinates_array(all_descr_df)
    distances = get_distances(coord_array)

    path_contexts = dict()
    path_contexts['distances'] = distances
    path_contexts['prompt'] = response.content

    final_prompt = prompt_paths.format(**path_contexts)
    response = llm.invoke(final_prompt)

    try:
        return response.content
    except AttributeError:
        return response
       
def main():
    places_paths = ["./datasets/compiled_toy_hotels.csv", "./datasets/compiled_toy_cafes.csv", "./datasets/compiled_toy_entertainments.csv"]
    docs = []

    # загрузка данных для ретривера
    hotels_loader = CSVLoader(file_path=places_paths[0], content_columns=['description'],
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
        })
    hotels_docs = hotels_loader.load()
    hotels_docs = docs_cleaning(hotels_docs)

    cafes_loader = CSVLoader(file_path=places_paths[1], content_columns=['description'],
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
        })
    cafes_docs = cafes_loader.load()
    cafes_docs = docs_cleaning(cafes_docs)

    ent_loader = CSVLoader(file_path=places_paths[2], content_columns=['description'],
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
        })
    ent_docs = ent_loader.load()
    ent_docs = docs_cleaning(ent_docs)

    # разделяем текст на части
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=200, add_start_index=True
        )

    # создаем эмбеддинги для частей текста
    embeddings_generator = HuggingFaceInferenceAPIEmbeddings(api_key=HG_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2")

    # создаем объекты для поиска по векторному хранилищу
    hotels_retriever = make_retriver("hotels", hotels_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)

    ent_retriever = make_retriver("ents", ent_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)

    cafes_retriever = make_retriver("cafes", cafes_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)

    # шаблон запроса для языковой модели
    template = """Вы являетесь помощником при составлении маршрутов по Новосибирской области.
    Используйте информацию только из найденного контекста, чтобы ответить на вопрос. При вопросе об определенном городе предоставляйте места только из него. Указывайте название, адрес и цену мест. Не добавляйте свои данные. Отвечайте на русском, понятно, кратко и лаконично. Используйте слова из запроса пользователя.
    Данные о развлечениях Новосибирска: {context_ent}  
    Данные об отелях Новосибирска: {context_hotels}
    Данные о местах питания в Новосибирске: {context_cafes}
    Если вы не знаете ответа или у вас нет данных для корректного ответа на него, просто скажите, что не знаете. 
    Вопрос от пользователя: {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)

    template_paths = """Добавь в предложенный маршрут информацию о расстояниях между объектами, которые предложено посетить за день. Напишите примерное время в пути при передвижении пешком и на машине. Предупредите, что расстояния и время приблизительные и необходимо уточнить информацию в интернете. В ответе оставь только текст для пользователя.
    Предложенный маршрут: {prompt}  
    Данные о расстояниях между объектами: {distances}

    """
    prompt_paths = ChatPromptTemplate.from_template(template_paths)

    # инициализируем языковую модель
    llm = HuggingFaceEndpoint(repo_id ="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature = 0.7, max_new_tokens = 1024, huggingfacehub_api_token = HG_API_KEY, task="text-generation",cache=False)

    # цикл для взаимодействия с пользователем
    question = ""
    while question.lower() != "q":
        question = input("Задайте свой вопрос (или введите 'q' для выхода): ")
        if question.lower() != "q":
            print("Вы: ", question) 
            print("Бот: ", run_rag_chain(llm, question, prompt, prompt_paths, hotels_retriever, cafes_retriever, ent_retriever))

if __name__ == "__main__":
    main()