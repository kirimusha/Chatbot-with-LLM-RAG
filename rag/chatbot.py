from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class ChatBot:
    def __init__(self, vector_db, model_name="deepseek-r1:1.5b"):
        """
        Инициализация чат-бота с поддержкой разных моделей
        
        Параметры:
            vector_db - векторное хранилище
            model_name - название модели (по умолчанию deepseek-r1:1.5b)
        """
        self.db = vector_db
        self.model_name = model_name
        self.llm = self._init_llm()
        
        # Динамический промпт с учетом выбранной модели
        self.prompt_template = """
            Ты {model_name} - AI ассистент. Отвечай строго по контексту:
            
            Контекст:
            {context}
            
            Вопрос:
            {question}
            
            Требования:
            1. Отвечай точно по контексту
            2. Форматируй ответ в Markdown
            3. Если ответа нет в контексте, скажи об этом
            
            Ответ:
        """
        self.chain = self.build_chain()
    
    def _init_llm(self):
        config = {
            "temperature": 0.7, # balance between creativity and correctness
            "base_url": "http://localhost:11434"
        }
            
        return ChatOllama(model=self.model_name, **config)
    
    def build_chain(self):
        """Сборка цепочки для вопросно-ответной системы"""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
            partial_variables={"model_name": self.model_name}
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def qa(self, question):
        """
        Получение ответа на вопрос
        
        Параметры:
            question - текст вопроса
            
        Возвращает:
            dict: {"answer": текст ответа, "sources": список источников}
        """
        try:
            result = self.chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": list(set(
                    doc.metadata.get("source", "unknown") 
                    for doc in result["source_documents"]
                ))
            }
        except Exception as e:
            return {
                "answer": f"Ошибка ({self.model_name}): {str(e)}",
                "sources": []
            }