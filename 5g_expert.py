import os
import numpy as np
from pypdf import PdfReader
import ollama
import faiss
import gradio as gr

PDF_DIR = "Database"
DB_DIR = "Vector_embeddings_database/3GPP_Database"
MODEL_NAME = "my-qwen"
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
        print("Загружаю базу из файлов.")
        index = faiss.read_index(index_path)
        chunks = np.load(texts_path, allow_pickle=True).tolist()
        return index, chunks

    print("База не найдена. Создаю новую.")
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

index, all_chunks = create_or_load_db()

def get_context(query, index, chunks, k=2):
    resp = client.embeddings(model=EMBED_MODEL, prompt=query)
    query_emb = np.array([resp['embedding']]).astype('float32')
    distances, indices = index.search(query_emb, k)
    context_parts = [chunks[i] for i in indices[0] if i != -1 and i < len(chunks)]
    return "\n---\n".join(context_parts)

def chat_process(message, history):
    if index is None:
        yield "Ошибка: База данных не загружена. Положите PDF в папку Database."
        return

    context = get_context(message, index, all_chunks, k=2)
    
    raw_prompt = (
            f"<|im_start|>system\n"
            f"Ты — ведущий инженер по сетям 5G и эксперт по стандартам 3GPP. "
            f"Твоя цель: дать развернутый, понятный и технически грамотный ответ на русском языке. \n\n"
            f"ПРАВИЛА:\n"
            f"1. Используй предоставленную ДОКУМЕНТАЦИЮ как главный источник истины.\n"
            f"2. Пиши структурированно: используй заголовки, жирный шрифт для терминов и маркированные списки.\n"
            f"3. Если в тексте есть сложные аббревиатуры (например, AMF, gNB, PDU), кратко поясняй их значение.\n"
            f"4. Не используй теги <think>. Твой тон: профессиональный, дружелюбный, экспертный.\n"
            f"5. Если информации в документах недостаточно, ответь на основе того, что есть, но упомяни, что это согласно доступным фрагментам.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ:\n{context}\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    try:
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
            
            clean_output = full_response.split("</think>")[-1].strip()
            yield clean_output

    except Exception as e:
        yield f"Ошибка при генерации: {str(e)}"

gr.close_all()

demo = gr.ChatInterface(
    fn=chat_process,
    title="📡 5G Ассистент (RAG Система)",
    description="Локальный доступ к базе 3GPP через языковую модель Qwen3:4B.",
)

if __name__ == "__main__":
    gr.close_all()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=True
    )