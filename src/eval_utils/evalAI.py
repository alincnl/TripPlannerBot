import re

import pandas as pd
from langchain_core.messages import HumanMessage
from tqdm import tqdm


def generate_ai_metrics(llm, question, answer, rag_data):
    '''
        ИИ-оцениватель RAG-системы, которая помогает планировать туристические маршруты по Новосибирской области.
        
        Для каждой пары вопрос-ответ рассчитывает показатели: достоверность, безопасность, детальность, релевантность, учет бюджета, учет длительности, полезность, учет логистики.

        Параметры
        ---------
        llm : ChatOpenAI
            Соединение с LLM
        rag_data : str
            Документы, которые передаются RAG-системе в текстовом формате
        rag_testset : pandas.DataFrame
            Датасет, который содержит вопросы (question), которые задавали системе, и ответы системы на них (answer). Датасет может также содержать другие столбцы.

        Возвращает
        ----------
        str   
            Метрики в формате 'Параметр - Оценка', разделенные точкой с запятой.
    '''
     
    messages = [
        HumanMessage(
            content=f"Ты - специалист по оценке RAG-системы, которая помогает с планированием туристических маршрутов по Новосибирской области. В этих трех документах описаны места в Новосибирске: {rag_data}\n\n Есть вопрос от пользователя: {question}. И ответ, который дала система: {answer}. Тебе нужно оценить систему по след. параметрам: достоверность (все данные имеют подтверждение в документе), безопасность (ответ не несет вред и не грубит пользователю), детальность (есть название места, при необходимости адрес, цены и описание), соответствие вопросу (ответ отвечает на поставленный вопрос или честно говорит, что не знает ответа), учет заданного бюджета (подобранный маршрут укладывается в заданный пользователем бюджет, если бюджет в вопросе не был указан, ставь 5), учет заданной продолжительности (маршрут составлен с учетом продолжительности, заданной пользователем, если продолжительность не была указана, ставь 5), полезность ответа (подобранное место/ответ действительно пригодятся и подходят пользователю), логистика (ответ содержит данные о расстояниях между местами и примерное время в пути на машине и пешком). Каждый параметр оценивай по пятибалльной шкале. В ответе дай только параметры и их оценку в формате 'Параметр - Оценка', разделяй параметры знаком ;"
        )
    ]

    answer = llm.invoke(messages).content
    return answer

def generate_ai_metrics_no_truth(llm, question, answer, rag_data):
    '''
        ИИ-оцениватель RAG-системы, которая помогает планировать туристические маршруты по Новосибирской области.
        
        Для каждой пары вопрос-ответ рассчитывает показатели: достоверность, безопасность, детальность, релевантность, учет бюджета, учет длительности, полезность, учет логистики.

        Параметры
        ---------
        llm : ChatOpenAI
            Соединение с LLM
        rag_data : str
            Документы, которые передаются RAG-системе в текстовом формате
        rag_testset : pandas.DataFrame
            Датасет, который содержит вопросы (question), которые задавали системе, и ответы системы на них (answer). Датасет может также содержать другие столбцы.

        Возвращает
        ----------
        str   
            Метрики в формате 'Параметр - Оценка', разделенные точкой с запятой.
    '''
     
    messages = [
        HumanMessage(
            content=f"Ты - специалист по оценке RAG-системы, которая помогает с планированием туристических маршрутов по Новосибирской области. Есть вопрос от пользователя: {question}. И ответ, который дала система: {answer}. Тебе нужно оценить систему по след. параметрам: безопасность (ответ не несет вред и не грубит пользователю), детальность (есть название места, при необходимости адрес, цены и описание), соответствие вопросу (ответ отвечает на поставленный вопрос или честно говорит, что не знает ответа), учет заданного бюджета (подобранный маршрут укладывается в заданный пользователем бюджет, если бюджет в вопросе не был указан, ставь 5), учет заданной продолжительности (маршрут составлен с учетом продолжительности, заданной пользователем, если продолжительность не была указана, ставь 5), полезность ответа (подобранное место/ответ действительно пригодятся и подходят пользователю), логистика (ответ содержит данные о расстояниях между местами и примерное время в пути на машине и пешком). Каждый параметр оценивай по пятибалльной шкале. В ответе дай только параметры и их оценку в формате 'Параметр - Оценка', разделяй параметры знаком ;"
        )
    ]

    answer = llm.invoke(messages).content
    return answer

def get_metrics_for_each_question(llm, rag_data, rag_testset, truth=True):
    """
    ИИ-оцениватель RAG-системы, которая помогает планировать туристические маршруты по Новосибирской области.
    
    Для каждой пары вопрос-ответ рассчитывает показатели: достоверность, безопасность, детальность, релевантность, учет бюджета, учет длительности, полезность, учет логистики.

    Параметры
    ---------
    llm : ChatOpenAI
        Соединение с LLM
    rag_data : str
        Документы, которые передаются RAG-системе в текстовом формате
    rag_testset : pandas.DataFrame
        Датасет, который содержит вопросы (question), которые задавали системе, и ответы системы на них (answer). Датасет может также содержать другие столбцы.

    Возвращает
    ----------
    pandas.DataFrame   
        Датасет с метриками по каждому вопросу-ответу.
    """

    ai_metrics = []

    for i in tqdm(range(len(rag_testset))):
        if truth:
            cur_metrics = generate_ai_metrics(llm, rag_testset.iloc[i]['question'], rag_testset.iloc[i]['answer'], rag_testset.iloc[i]['contexts'])
        else:
            cur_metrics = generate_ai_metrics_no_truth(llm, rag_testset.iloc[i]['question'], rag_testset.iloc[i]['answer'], rag_data)

        # парсинг ответа модели
        numbers = re.findall(r'\d+', cur_metrics)
        ai_metrics.append(list(map(int, numbers)))

    if truth:
        metrics_df = pd.DataFrame(ai_metrics, columns=['truth', 'safety', 'detail','relevance', 'price', 'duration', 'utility', 'logistics'])
    else:
        metrics_df = pd.DataFrame(ai_metrics, columns=['safety', 'detail','relevance', 'price', 'duration', 'utility', 'logistics'])

    metrics_df.insert(loc=0, column='question', value=rag_testset['question'])
    metrics_df.insert(loc=1, column='answer', value=rag_testset['answer'])

    return metrics_df

def get_mean_metrics(llm, rag_data, rag_testset, truth=True):
    """
    ИИ-оцениватель RAG-системы, которая помогает планировать туристические маршруты по Новосибирской области.
    
    Рассчитывает средние по датасету показатели: достоверность, безопасность, детальность, релевантность, учет бюджета, учет длительности, полезность, учет логистики.

    Параметры
    ---------
    llm : ChatOpenAI
        Соединение с LLM
    rag_data : str
        Документы, которые передаются RAG-системе в текстовом формате
    rag_testset : pandas.DataFrame
        Датасет, который содержит вопросы (question), которые задавали системе, и ответы системы на них (answer). Датасет может также содержать другие столбцы.

    Возвращает
    ----------
    dict  
        JSON с усредненными по датасету метриками.
    """

    ai_metrics = []

    for i in tqdm(range(len(rag_testset))):
        if truth:
            cur_metrics = generate_ai_metrics(llm, rag_testset.iloc[i]['question'], rag_testset.iloc[i]['answer'],  rag_testset.iloc[i]['contexts'])
        else:
            cur_metrics = generate_ai_metrics_no_truth(llm, rag_testset.iloc[i]['question'], rag_testset.iloc[i]['answer'], rag_data)

        # парсинг ответа модели
        numbers = re.findall(r'\d+', cur_metrics)
        ai_metrics.append(list(map(int, numbers)))

    if truth:
        metrics_df = pd.DataFrame(ai_metrics, columns=['truth', 'safety', 'detail','relevance', 'price', 'duration', 'utility', 'logistics'])
    else:
        metrics_df = pd.DataFrame(ai_metrics, columns=['safety', 'detail','relevance', 'price', 'duration', 'utility', 'logistics'])

    if truth:
        mean_truth = metrics_df['truth'].mean()
    mean_safety = metrics_df['safety'].mean()
    mean_detail = metrics_df['detail'].mean()
    mean_relevance = metrics_df['relevance'].mean()
    mean_utility = metrics_df['utility'].mean()
    mean_price = metrics_df['price'].mean()
    mean_duration = metrics_df['duration'].mean()
    mean_logistics = metrics_df['logistics'].mean()
    
    if truth:
        metrics_json =  {'Достоверность': mean_truth, 'Безопасность': mean_safety, 'Детальность': mean_detail, 'Релевантность': mean_relevance, 'Учет цены': mean_price, 'Учет продолжительности': mean_duration, 'Полезность': mean_utility, 'Логистика': mean_logistics}
    else:
        metrics_json =  {'Безопасность': mean_safety, 'Детальность': mean_detail, 'Релевантность': mean_relevance, 'Учет цены': mean_price, 'Учет продолжительности': mean_duration, 'Полезность': mean_utility, 'Логистика': mean_logistics}

    return metrics_json