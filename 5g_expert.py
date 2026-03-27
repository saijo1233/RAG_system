import os
import numpy as np
from pypdf import PdfReader
import ollama
import faiss
import gradio as gr

# --- Настройки ---
PDF_DIR = "Database"
DB_DIR = "Vector_embeddings_database/3GPP_Database"
MODEL_NAME = "my-qwen" # Убедитесь, что это имя совпадает с !ollama list
EMBED_MODEL = "bge-m3"

client = ollama.Client(host='http://127.0.0.1:11434')

def get_pdf_chunks(path):
    if not os.path.exists(path): return []
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text = " ".join(text.split())
            for i in range(0, len(text), 700):
                chunk = text[i:i+800].strip()
                if len(chunk) > 50:
                    chunks.append(chunk)
    return chunks

def create_or_load_db():
    if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
    index_path = os.path.join(DB_DIR, "index.faiss")
    texts_path = os.path.join(DB_DIR, "texts.npy")

    if os.path.exists(index_path) and os.path.exists(texts_path):
        print("--- Загружаю базу из файлов... ---")
        index = faiss.read_index(index_path)
        chunks = np.load(texts_path, allow_pickle=True).tolist()
        return index, chunks

    print("--- База не найдена. Создаю новую... ---")
    chunks = []
    if not os.path.exists(PDF_DIR): os.makedirs(PDF_DIR, exist_ok=True)
    
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            chunks.extend(get_pdf_chunks(os.path.join(PDF_DIR, filename)))

    if not chunks: return None, []

    embeddings_list = [client.embeddings(model=EMBED_MODEL, prompt=txt)['embedding'] for txt in chunks]
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(len(embeddings_list[0]))
    index.add(embeddings_matrix)
    
    faiss.write_index(index, index_path)
    np.save(texts_path, np.array(chunks))
    return index, chunks

# --- Инициализация базы ---
index, all_chunks = create_or_load_db()

def get_context(query, index, chunks, k=2):
    resp = client.embeddings(model=EMBED_MODEL, prompt=query)
    query_emb = np.array([resp['embedding']]).astype('float32')
    distances, indices = index.search(query_emb, k)
    context_parts = [chunks[i] for i in indices[0] if i != -1 and i < len(chunks)]
    return "\n---\n".join(context_parts)

# --- Функция для чата Gradio ---
def chat_process(message, history):
    if index is None:
        yield "Ошибка: База данных не загружена. Положите PDF в папку Database."
        return

    # 1. Поиск контекста
    context = get_context(message, index, all_chunks, k=2)
    
    # 2. Формирование промпта
    raw_prompt = (
        f"<|im_start|>system\n"
        f"Ты эксперт по 5G. Отвечай на русском языке, используя ТОЛЬКО предоставленную документацию. "
        f"ЗАПРЕЩЕНО использовать теги <think>. Пиши кратко и по существу.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"ДОКУМЕНТАЦИЯ:\n{context}\n\n"
        f"ВОПРОС: {message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        # 3. Стриминг ответа
        full_response = ""
        stream = client.generate(
            model=MODEL_NAME,
            prompt=raw_prompt,
            stream=True,
            options={"num_ctx": 4096, "temperature": 0.2, "stop": ["<|im_start|>", "<|im_end|>"]}
        )

        for chunk in stream:
            content = chunk.get('response', '')
            full_response += content
            
            # Очистка от возможных остатков тегов рассуждения
            clean_output = full_response.split("</think>")[-1].strip()
            yield clean_output

    except Exception as e:
        yield f"Ошибка при генерации: {str(e)}"

# --- Интерфейс Gradio ---
gr.close_all() # Закрываем старые сессии

demo = gr.ChatInterface(
    fn=chat_process,
    title="📡 5G Expert Assistant",
    description="Задайте вопрос по документации 3GPP. Система использует RAG (FAISS + Ollama).",
)

if __name__ == "__main__":
    print("--- Попытка запуска... ---")
    # Оставляем только базовые параметры: 
    # share=True для внешней ссылки, debug=True для логов
    demo.launch(
        share=True, 
        debug=True
    )