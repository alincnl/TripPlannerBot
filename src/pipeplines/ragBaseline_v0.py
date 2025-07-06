import requests
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    # указываем URL страницы с данными и загружаем с нее данные
    page = "https://trip2sib.ru/putevoditel/belovskiy-vodopad/"
    loader = WebBaseLoader(page)
    docs = loader.load() 
    
    # разделяем текст на части
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Максимальный размер части текста
            chunk_overlap=200,  # Перекрытие между частями
            add_start_index=True  # Добавляем индекс начала
        )
    splits = text_splitter.split_documents(docs)
    
    # создаем эмбеддинги для частей текста
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HG_API_KEY, 
        model_name="sentence-transformers/all-MiniLM-l6-v2"  # Указываем модель для получения эмбеддингов
    )
    
    # создаем векторное хранилище на основе эмбеддингов
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)

    # создаем объект для поиска по векторному хранилищу
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # шаблон запроса для языковой модели
    template = """Вы являетесь помощником при выполнении заданий, связанных с ответами на вопросы. 
    Используйте следующие фрагменты найденного контекста и свои знания, чтобы ответить на вопрос. 
    Используйте не более трех предложений и будьте лаконичны в ответе. Удаляйте ведущие и конечные пробелы.
    Вопрос: {question} 
    Контекст: {context} 
    Ответ:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # объединение содержимого документов в одну строку
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # инициализируем языковую модель
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        temperature=0.5,
        max_new_tokens=128,
        huggingfacehub_api_token=HG_API_KEY
    )

    # создаем цепочку для обработки запросов
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  # извлечение контекста и вопроса
        | prompt  # применение шаблона
        | llm  # генерация ответа с помощью языковой модели
        | StrOutputParser()  # парсинг строки с ответом
    )

    # цикл для взаимодействия с пользователем
    question = ""
    while question.lower() != "q":
        question = input("Задайте свой вопрос (или введите 'q' для выхода): ")
        if question.lower() != "q":
            print("Вы: ", question) 
            print("Бот: ", rag_chain.invoke(question))

if __name__ == "__main__":
    main()
