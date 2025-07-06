import json
import logging
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from geoloc_utils.ragGeocoder import get_coordinates_array, get_distances
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            HuggingFaceInferenceAPIEmbeddings)
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Игнорирование предупреждений
warnings.filterwarnings("ignore")

log_dir = Path("src/logs")
log_dir.mkdir(exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'tg_logs_{current_date}.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка данных и создание ретриверов
places_paths = [
    "./src/production_data/production_hotels_districts.csv",
    "./src/production_data/production_cafes_districts.csv",
    "./src/production_data/production_ent_districts.csv"
]
docs = []

def docs_cleaning(docs):
    for i in range(len(docs)):
        cleaned_content = re.sub(r'(: [0-9+]\\n0:)', ' ', docs[i].page_content).strip()
        docs[i].page_content = cleaned_content
        docs[i].page_content = docs[i].page_content.replace("description: ", "")
    return docs

def create_loader(file_path):
    loader = CSVLoader(file_path=file_path, content_columns=['description'],
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
    })
    docs = loader.load()
    docs = docs_cleaning(docs)
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_retriver(name: str, documents: list, embedding_generator, text_splitter, search_type: str = "mmr", search_kwargs: dict = {"k": 5}):
    text_splits = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(collection_name=name, documents=text_splits, embedding=embedding_generator, persist_directory="/tmp/chroma_db")
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


# Загрузка документов
hotels_docs = create_loader(places_paths[0])
cafes_docs = create_loader(places_paths[1])
ent_docs = create_loader(places_paths[2])

# Разделение текста на части
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100, add_start_index=True
)
splits = text_splitter.split_documents(docs)

# Создание эмбеддингов
embeddings_generator = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
)

hotels_retriever = make_retriver("hotels", hotels_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)
ent_retriever = make_retriver("ents", ent_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)
cafes_retriever = make_retriver("cafes", cafes_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)

if hasattr(hotels_retriever, 'vectorstore'):
    hotels_retriever.vectorstore.delete_collection()
if hasattr(ent_retriever, 'vectorstore'):
    ent_retriever.vectorstore.delete_collection()
if hasattr(cafes_retriever, 'vectorstore'):
    cafes_retriever.vectorstore.delete_collection()

hotels_retriever = make_retriver("hotels", hotels_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)
ent_retriever = make_retriver("ents", ent_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)
cafes_retriever = make_retriver("cafes", cafes_docs, embedding_generator=embeddings_generator, text_splitter=text_splitter)
                                
# Инициализация модели LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=API_KEY,
    base_url=PROXY_URL,
)

# Шаблоны промптов
template_main = """Вы являетесь дружелюбным помощником при составлении маршрутов по Новосибирской области, но можете немного и поговорить о жизни.
Используйте информацию только из найденного контекста, чтобы ответить на вопрос. При вопросе об определенном городе предоставляйте места только из него. Указывайте название, адрес (без учета района, если не просят обратное), краткое описание (если оно есть) и цену мест. Не добавляйте свои данные. Отвечайте на русском, понятно, кратко и лаконично. Используйте слова из запроса пользователя.
При составлении маршрута на 1 день и более ОБЯЗАТЕЛЬНО указывайте места, где можно покушать и где переночевать. Указывай одно место для ночлега на все дни, если не просят обратного. Если попросят порекомендовать что-то рядом, то выбирайте места, которые находятся В ОДНОМ РАЙОНЕ, если мест в одном районе нет или вам неизвестен район - предупреждайте, что место не рядом 
Данные о развлечениях и санаториях Новосибирска: {context_ent}  
Данные об отелях и санаториях Новосибирска: {context_hotels}
Данные о местах питания в Новосибирске: {context_cafes}
Учитывайте историю диалога и старайтесь не допускать повтора мест при повторной просьбе от пользователя.
Если вы не знаете ответа или у вас нет данных для корректного ответа на него, просто скажите, что не знаете. 
Вопрос от пользователя: {question} 
"""
prompt = ChatPromptTemplate.from_template(template_main)

template_paths = """Выполни следующие шаги:
1. Если текст от бота УЖЕ содержит информацию о расстояниях между местами в маршруте (например, "от такого место до такого расстояние составляет", "км", "путь займет минут"), то ВЕРНИ ИСХОДНЫЙ ТЕКСТ ДЛЯ ПОЛЬЗОВАТЕЛЯ БЕЗ ИЗМЕНЕНИЙ.
2. Если текст НЕ является маршрутом или планом мест для путешествия/свидания/отдыха и др., ВЕРНИ ИСХОДНЫЙ ДЛЯ ПОЛЬЗОВАТЕЛЯ ТЕКСТ БЕЗ ИЗМЕНЕНИЙ.
3. Если текст содержит список различных мест и НЕ содержит информации о расстояниях, то
  - добавь информацию о расстояниях ТОЛЬКО между теми объектами, которые УЖЕ упомянуты в тексте от бота
  - укажи примерное время в пути пешком и на машине
  - добавь предупреждение о том, что расстояния и время приблизительные. 
В ответе оставь только текст для пользователя.
Текст от бота: {prompt}
Данные о расстояниях между объектами: {distances}
"""
prompt_paths = ChatPromptTemplate.from_template(template_paths)

def get_retriver_categories(llm, question):
    '''template = """Есть вопрос от пользователя: {question} 
    Оцени, будет ли пользователю полезна для ответа на вопрос информация о: местах проживания, местах питания и местах для проведения досуга. Чтобы определить надобность, не додумывай вопрос пользователя, отвечай только по тому вопросу, что спрашивается. Однако если просят составить маршрут или план на 1 день и более, то отвечай ВСЕГДА [true, true, true]. Еще отвечай ВСЕГДА [true, true, true], если просят узнать о расстоянии между объектами или спрашивают адрес чего-то.
    В ответе напиши список из true и false, где на первом месте будет надобность мест проживания, на втором - мест питания (рестораны, кафе и прочее) и на третьем - развлечения (бассейны, парки, музеи и прочее). К примеру, [false, false, true].
    """'''
    template = """Анализируя вопрос пользователя, определи, какая информация ему нужна. Строго следуй правилам:

    ПРАВИЛА ОЦЕНКИ:
    1. Если вопрос содержит просьбу о:
    - маршруте/плане на 1+ дней → [true, true, true]
    - расстоянии между объектами → [true, true, true]
    - адресе чего-либо → [true, true, true]
    - "где остановиться" → [true, false, false]
    - "где поесть" → [false, true, false]
    - "куда сходить" → [false, true, true]

    2. Для общих вопросов о городе/месте:
    - "что посмотреть" → [false, false, true]
    - "что интересного" → [false, false, true]
    
    3. НЕ ДОДУМЫВАЙ. Если вопрос не подходит под правила - верни [false, false, false]

    ФОРМАТ ОТВЕТА:
    Только JSON-массив вида [проживание, питание, досуг], например:
    - "Где можно поесть?" → [false, true, false]
    - "Составь маршрут ..." → [true, true, true]
    - "Как добраться до..?" → [true, true, true]
    - "Где находится..?" → [true, true, true]
    - "Как дела?" → [false, false, false]
    - "Хочу торт (или любой другой десерт/напиток/блюдо)" → [false, true, false]
    - "Хочу в " → [true, true, true]

    Вопрос: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    result = llm.invoke(prompt.format(question=question))

    try:
        result = result.content
    except AttributeError:
        result = result

    pattern = re.compile(r"\[.*\]")
    match = pattern.search(result)
    try:
        return json.loads(match[0])
    except:
        return [True, True, True]

class RawContext:
    def __init__(self, context_name: str = "Unknown", 
                 retriever: VectorStoreRetriever = None, 
                 context_default_output: str = "нет необходимости"):
        self.context_name = context_name
        self.retriever = retriever
        self.context_default_output = context_default_output

def get_contexts_(username, llm, question: str, raw_contexts: list[RawContext]):
    categories_bool_list = get_retriver_categories(llm, question)
   # print("Получил список категорий поиска (отели, кафе, развлечения): ", categories_bool_list)
    logger.info(f"User: {username}, context results (hotels, cafes, ents): {categories_bool_list}")

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

def run_rag_chain(username, question, prompt, prompt_paths, history="", retr_question="", history_coord_arr=[]):
    logger.info(f"User: {username}, Action: starting RAG chain, Question: {question}")
    context_question = question if history == "" else retr_question
    if ("рядом с новос" in question.lower() or "недалеко с новос" in question.lower()):
        context_question = context_question.lower().replace("новосибирск", "")
        context_question += "например, в Бердске, Кольцово, Колывани"
    
    contexts = get_contexts_(username, llm, context_question, raw_contexts=[
        RawContext(context_name="context_hotels", retriever=hotels_retriever),
        RawContext(context_name="context_cafes", retriever=cafes_retriever),
        RawContext(context_name="context_ent", retriever=ent_retriever)
    ])

    if retr_question != "":
        contexts['question'] = question
    logger.info(f"User: {username}, Action: processing contexts")
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
    #print("Полученный контекст: ", all_descr_df)
    final_prompt = prompt.format(**(contexts))
    if history != "":
        final_prompt += "\nИстория диалога: " + history
    
    response = llm.invoke(final_prompt)
    search_text = response.content.lower() + question
    #print("Составил первоначальный ответ")
    logger.info(f"User: {username}, Action: generating initial response")

    if (("маршрут" not in search_text) and ("план" not in search_text) and ("поездк" not in search_text) and ("путь" not in search_text) and ("пути" not in search_text) and ("путешеств" not in search_text) and ("рядом" not in search_text) and ("недалеко" not in search_text) and ("расст" not in search_text)) or (empty_cat>=2):
        #print("Не увидел надобности считать расстояния")
        logger.info(f"User: {username}, Action: skipping distance calculation")
        return response.content, []
    
    coord_array = get_coordinates_array(all_descr_df)
    distances = get_distances(coord_array + history_coord_arr)

    path_contexts = dict()
    path_contexts['distances'] = distances
    path_contexts['prompt'] = response.content
    #print("Идем считать расстояния с таким запросом: ", response.content)
    logger.info(f"User: {username}, Action: calculating distances")
    final_prompt = prompt_paths.format(**path_contexts)
    response = llm.invoke(final_prompt)
    #print("Готово")
    logger.info(f"User: {username}, Action: response generation complete")
    try:
        return response.content, coord_array
    except AttributeError:
        return response, coord_array

@dataclass
class ChatMessage:
    role: str  # "user" или "assistant"
    content: str

class DialogManager:
    def __init__(self):
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.retriever_prompt = (
            "Просто переформулируй последнее сообщение user, чтобы его посыл был понятен досконально без всей остальной истории. Если последнее сообщение не нуждается в переформулировке, то напиши его, какое оно есть. НЕ ВЗДУМАЙ ОТВЕЧАТЬ НА СООБЩЕНИЕ"
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        self.coord_retr_array = []
    
    def _log_generation(self, username: str, action: str, details: str = ""):
        """Логирование процесса генерации"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] User: {username}, Action: {action}"
        if details:
            log_message += f", Details: {details}"
        logger.info(log_message)

    def get_session(self, username: str) -> List[ChatMessage]:
        """Возвращает или создает сессию для пользователя"""
        if username not in self.sessions:
            self.sessions[username] = []
        return self.sessions[username]
    
    def add_message(self, username: str, role: str, content: str):
        """Добавляет сообщение в историю чата пользователя"""
        session = self.get_session(username)
        session.append(ChatMessage(role=role, content=content))
    
    def get_chat_history(self, username: str) -> List[ChatMessage]:
        """Возвращает историю чата пользователя"""
        return self.get_session(username)
    
    def clear_history(self, username: str):
        """Очищает историю чата пользователя"""
        self.sessions[username] = []

    def get_user_questions(self, chat_history: List[ChatMessage]) -> List[str]:
        """Возвращает только вопросы пользователя из истории"""
        return [msg.content for msg in chat_history if msg.role == "user"]
    
    def format_chat_history_as_str(self, chat_history: List[ChatMessage]) -> str:
        """Форматирует историю чата в строку"""
        history_str = ""
    
        # Сначала собираем всю историю
        for msg in chat_history:
            role = "Пользователь" if msg.role == "user" else "Бот"
            history_str += f"{role}: {msg.content}\n"
        
        # Если история слишком длинная
        if len(history_str) > 1500:
            # Оставляем последние 4 сообщения полностью
            last_messages = []
            total_length = 0
            for msg in reversed(chat_history):
                msg_str = f"{'Пользователь' if msg.role == 'user' else 'Бот'}: {msg.content}\n"
                if total_length + len(msg_str) > 1500:
                    break
                last_messages.append(msg_str)
                total_length += len(msg_str)
            
            # Собираем обрезанную историю (новые сообщения в конце)
            history_str = "".join(reversed(last_messages))
            
            # Добавляем пометку о том, что история была обрезана
            history_str = "[Часть истории была обрезана]\n" + history_str
        return history_str.strip()


    def generate_response(self, username: str, user_message: str, prompt, prompt_paths) -> str:
        """Генерирует ответ с учетом истории чата"""
        self.add_message(username, "user", user_message)
        chat_history = self.get_chat_history(username)
        user_questions = self.get_user_questions(chat_history[:-1])
        messages_for_prompt = [("user", q) for q in user_questions]
        
        try:
            contextualized_question_prompt = self.contextualize_q_prompt.format_messages(
                chat_history=messages_for_prompt,
                input=user_message
            )
            contextualized_question = llm.invoke(contextualized_question_prompt)
            contextualized_question = contextualized_question.content if hasattr(contextualized_question, 'content') else contextualized_question
        except Exception as e:
            #print(f"Ошибка при контекстуализации вопроса: {e}")
            logger.info(f"User: {username}, contextulaization error: {e}")
            contextualized_question = user_message
            
        full_history_str = self.format_chat_history_as_str(chat_history)
        response, coord_array = run_rag_chain(username, user_message, prompt, prompt_paths, full_history_str, contextualized_question, self.coord_retr_array)
        #print("Длина истории: ", len(full_history_str))
        self.coord_retr_array = coord_array
        self.add_message(username, "assistant", response)
        return response
    
def main():
    print("Напишите свой вопрос. \n")
    print("Для выхода введите 'выход' или 'exit'.\n")
    
    dialog_manager = DialogManager()
    username = "console_user"
    
    while True:
        try:
            user_input = input("Вы: ")
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
                
            if not user_input.strip():
                print("Пожалуйста, введите непустой запрос.")
                continue
                
            response = dialog_manager.generate_response(
                username=username,
                user_message=user_input,
                prompt=prompt,
                prompt_paths=prompt_paths
            )
            
            print("\nБот:", response)
            print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            continue

if __name__ == "__main__":
    main()