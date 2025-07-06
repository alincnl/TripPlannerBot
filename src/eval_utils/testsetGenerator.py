import argparse
import io
import re

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

API_KEY= 'YOUR_API_KEY'
PROXY_URL = 'https://bothub.chat/api/v2/openai/v1'

# достаем описание всех мест в тектсовом формате
def read_csv_as_text(filepath):
    buffer = io.StringIO()
    pd.read_csv(filepath, usecols=['description'],index_col=0).to_csv(buffer)
    buffer.seek(0)
    return buffer.read()

# генерация вопросов
def get_questions(llm):
    messages = [
        HumanMessage(
            content="Пишу RAG-систему, которая будет помогать с планированием туристических маршрутов по Новосибирской области: составлять маршруты с учетом интересов, бюджета и продолжительности отдыха, а иногда также отвечать на вопросы об определенных местах. Например: 'Составь маршрут по Новосибирску на 2 дня с посещением музеев и бюджетом 10000р'. Мне нужен тестовый датасет с возможными вопросами от пользователей, напиши, пожалуйста, 30 вопросов, которые содержат в себе лимит по цене, диапазон продолжительности отдыха и желаемые для посещения категории мест (небольшая часть вопросов может содержать только один или несколько параметров: к примеру, цена и места) в формате csv. В ответе передай только вопросы в формате csv"
        )
    ]

    answer = llm.invoke(messages)
    #print(answer)
    # достаем вопросы из ответа
    questions = re.findall(r'"(.*?)"', answer.content)
    questions = [item for item in questions if item != 'Вопрос']
    return questions

# получаем подходящие места из БД для каждого вопроса
def get_right_context(llm, question, csv_text):
    messages = [
        HumanMessage(
            content=f"В этих трех документах описаны места в Новосибирске: {csv_text}\n\n Есть вопрос от пользователя: {question}. Оставь только те места из документов в формате csv ровно так, как они записаны в документе, которые могут быть полезны при ответе на вопрос. Напиши эти места именно так, в каком виде они даны. К примеру, 'Мини-отель Роял Марин по адресу микрорайон Северный, д.1/1, Бердск. Минимальная стоимость:  'Мини-отель Роял Марин' за ночь от 3466₽'. Если нет подходящих мест, пиши фразу 'Недостаточно информации'."
        )
    ]

    answer = llm.invoke(messages)

    places = re.findall(r'"(.*?)"', answer.content)
    places = [item for item in places]
    return places

# python ./src/eval_utils/testsetGenerator.py --hotels ./src/production_data/production_hotels.csv --cafes ./src/production_data/production_cafes.csv --entertainments ./src/production_data/production_ent.csv

def main():
    parser = argparse.ArgumentParser(description="Объединение содержимого CSV-файлов в одну строку.")
    parser.add_argument("--hotels", type=str, required=True, help="Путь к CSV-файлу с отелями")
    parser.add_argument("--cafes", type=str, required=True, help="Путь к CSV-файлу с кафе")
    parser.add_argument("--entertainments", type=str, required=True, help="Путь к CSV-файлу с развлечениями")
    
    # Парсинг аргументов
    args = parser.parse_args()

    # Чтение и объединение CSV-файлов
    csv_text = (
        read_csv_as_text(args.hotels) + "\n\n" +
        read_csv_as_text(args.cafes) + "\n\n" +
        read_csv_as_text(args.entertainments)
    )

    # подключение к модели
    custom_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=API_KEY,
        base_url=PROXY_URL,
        cache=False
    )

    # получение вопросов и подбор для каждого необходимый контекст
    questions = get_questions(custom_llm)
    #print(questions)
    contexts = []
    for i in range(len(questions)):
       contexts.append(get_right_context(custom_llm, questions[i], csv_text))

    # сохраняем
    testset = {
        "question": questions,
        "reference_contexts": contexts
    }
    df = pd.DataFrame(testset)
    df.to_csv("./src/production_data/prod_testset_for_eval.csv")
 
if __name__ == "__main__":
    main()