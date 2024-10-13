import os
import pickle
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import util, SentenceTransformer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from openai import OpenAI

# Загрузка переменных окружения
load_dotenv(find_dotenv())

TELEGRAM_TOKEN = "YOUR TOKEN"

# Настройки LM Studio API
BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("API_KEY", "meta-llama-3.1-8b-instruct")

# Инициализация клиента OpenAI для LM Studio
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Инициализация модели SentenceTransformer
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Функция для загрузки векторного пространства
def load_vector_space(file_path='vector_space.pkl'):
    with open(file_path, 'rb') as f:
        sentences, sentence_embeddings = pickle.load(f)
    return sentences, sentence_embeddings

# Функция для нахождения релевантных предложений
def find_relevant_sections(question, sentences, sentence_embeddings, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)
    relevant_sentences = sorted(zip(sentences, scores[0]), key=lambda x: x[1], reverse=True)
    return relevant_sentences

# Функция для создания промпта
def create_prompt(question, vector_space_file='vector_space.pkl'):
    sentences, sentence_embeddings = load_vector_space(vector_space_file)
    relevant_sentences = find_relevant_sections(question, sentences, sentence_embeddings, semantic_model)[:5]
    context = " ".join([sent[0] for sent in relevant_sentences])
    return f"Текст: {context}\n\nВопрос: {question}\nОтвет:"

# Функция для отправки запроса к модели LM Studio
def send_prompt_to_model(prompt):
    response = client.chat.completions.create(
        model="model-identifier",  # Укажите идентификатор модели
        messages=[{"role": "user", "content": prompt}]
    )
    if response.choices:
        return response.choices[0].message.content
    else:
        return "Ошибка: нет ответа от модели."

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Задайте свой вопрос, и я постараюсь найти ответ.")

# Обработчик сообщений от пользователя
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    prompt = create_prompt(question, vector_space_file=r'C:\Users\fff\Desktop\NN\vector_space.pkl')
    answer = send_prompt_to_model(prompt)
    await update.message.reply_text(answer)

# Основная функция для запуска бота
def main():
    # Инициализация приложения Telegram
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрация обработчиков команд и сообщений
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    print("Бот запущен...")
    app.run_polling()

# Запуск приложения
if __name__ == "__main__":
    main()
