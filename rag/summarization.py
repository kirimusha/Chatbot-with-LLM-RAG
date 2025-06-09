from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class WebSummarizer:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        """
        Инициализация суммаризатора с поддержкой разных моделей
        
        Параметры:
            model_name - название модели (по умолчанию deepseek-r1:1.5b)
        """
        self.model_name = model_name
        self.llm = self._init_llm()
        
        # Промпт с учетом особенностей разных моделей
        self.prompt_template = """
            Ты {model_name} - эксперт по суммаризации. Создай краткое изложение:
            
            Требования:
            1. Выдели 3-5 ключевых пунктов
            2. Сохрани важные технические детали
            3. Используй Markdown-разметку
            4. Объем: 150-200 слов
            
            Контент:
            {content}
            
            Краткое изложение:
        """
    
    def _init_llm(self):
        config = {
            "temperature": 0.2,  # its should be strict and precise
            "base_url": "http://localhost:11434",
            "max_tokens": 500
        }
            
        return ChatOllama(model=self.model_name, **config)
    
    def summarize(self, content):
        """
        Создание краткого изложения контента
        
        Параметры:
            content - текст для суммаризации
            
        Возвращает:
            str: краткое изложение в Markdown
        """
        try:
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["content"],
                partial_variables={"model_name": self.model_name}
            )
            
            # Ограничение длины контента
            truncated_content = content[:15000]  # ~15k tokens
            
            response = self.llm.invoke([{
                "role": "user",
                "content": prompt.format(content=truncated_content)
            }])
            
            return response.content
        except Exception as e:
            return f"Ошибка суммаризации ({self.model_name}): {str(e)}"