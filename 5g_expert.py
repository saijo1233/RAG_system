import os
import numpy as np
from pypdf import PdfReader
import ollama
import faiss

PDF_DIR = "Database"
DB_DIR = "Vector_embeddings_database/3GPP_Database"
MODEL_NAME = "my-qwen"
EMBED_MODEL = "bge-m3"

client = ollama.Client(host='http://127.0.0.1:11434')

def get_pdf_chunks(path):
    print(f"--- Читаю PDF: {path} ---")
    if not os.path.exists(path):
        print(f"Ошибка: Файл {path} не найден!")
        return []
    
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
    if not os.path.exists(PDF_DIR):
        print(f"Внимание: Папка '{PDF_DIR}' не найдена. Создаю её...")
        os.makedirs(PDF_DIR, exist_ok=True)
        print(f"Пожалуйста, поместите PDF-файлы в папку '{PDF_DIR}' и перезапустите скрипт.")
        return None, []

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(PDF_DIR, filename)
            chunks.extend(get_pdf_chunks(filepath))

    if not chunks:
        print(f"Ошибка: Не найдено PDF-файлов в папке '{PDF_DIR}' или в них отсутствует текст.")
        return None, []

    print(f"--- Генерация векторов для {len(chunks)} фрагментов (может занять время)... ---")
    embeddings_list = []
    for i, txt in enumerate(chunks):
        resp = client.embeddings(model=EMBED_MODEL, prompt=txt)
        embeddings_list.append(resp['embedding'])
        if i % 50 == 0: print(f"Прогресс: {i}/{len(chunks)}")

    embeddings_matrix = np.array(embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(len(embeddings_list[0]))
    index.add(embeddings_matrix)
    
    faiss.write_index(index, index_path)
    np.save(texts_path, np.array(chunks))
    print(f"--- База создана! Чанков: {len(chunks)} ---")
    return index, chunks

def get_context(query, index, chunks, k=2):
    """ Находит k самых похожих фрагментов текста """
    resp = client.embeddings(model=EMBED_MODEL, prompt=query)
    query_emb = np.array([resp['embedding']]).astype('float32')
    distances, indices = index.search(query_emb, k)
    
    context_parts = []
    for i in indices[0]:
        if i != -1 and i < len(chunks):
            context_parts.append(chunks[i])
    return "\n---\n".join(context_parts)

if __name__ == "__main__":
    index, all_chunks = create_or_load_db()
    if not index:
        print("Ошибка: Не удалось инициализировать базу данных.")
        exit()

    print("\n--- СИСТЕМА ГОТОВА (введите 'выход' для завершения) ---")

    while True:
        user_q = input("\nВопрос: ")
        if user_q.lower() in ['выход', 'exit', 'quit']: break
        
        context = get_context(user_q, index, all_chunks, k=2)
        
        raw_prompt = (
            f"<|im_start|>system\n"
            f"Ты эксперт по 5G. Твоя задача: ответить на вопрос пользователя, используя ТОЛЬКО предоставленную документацию. "
            f"ЗАПРЕЩЕНО использовать теги <think> и рассуждать вслух. Пиши только финальный результат на русском языке.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"ДОКУМЕНТАЦИЯ:\n{context}\n\n"
            f"ВОПРОС: {user_q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        print("\nДумаю...", end="", flush=True)
        
        try:
            stream = client.generate(
                model=MODEL_NAME,
                prompt=raw_prompt,
                stream=True,
                options={
                    "num_ctx": 4096,      
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "stop": ["<|im_start|>", "<|im_end|>", "system\n", "user\n", "ВОПРОС:", "/<think>"] 
                }
            )

            full_response = ""
            for chunk in stream:
                content = chunk['response']
                full_response += content

            if "</think>" in full_response:
                clean_answer = full_response.split("</think>")[-1].strip()
            else:
                clean_answer = full_response.strip()

            print("\rОТВЕТ:", clean_answer)

            for chunk in stream:
                content = chunk['response']
                print(content, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"\nОшибка при генерации: {e}")