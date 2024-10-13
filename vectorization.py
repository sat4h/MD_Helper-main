import os
import nltk
from sentence_transformers import SentenceTransformer
import pickle

nltk.download('punkt_tab')


# Функция для получения всех .md файлов в папке
def get_md_files_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.md')]

# Функция для чтения markdown файлов
def read_markdown(file_paths):
    combined_text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            combined_text += f.read() + "\n"
    return combined_text

# Функция для разбиения текста на предложения
def split_text(text):
    return nltk.sent_tokenize(text)

# Функция для создания векторного пространства и сохранения в файл
def create_and_save_vector_space(directory, output_file='vector_space.pkl'):
    # Получаем все .md файлы из папки
    file_paths = get_md_files_from_directory(directory)
    
    # Чтение текстов из файлов
    large_text = read_markdown(file_paths)
    
    # Разбиваем текст на предложения
    sentences = split_text(large_text)
    
    # Загрузка модели для семантического анализа
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Векторизация предложений
    sentence_embeddings = semantic_model.encode(sentences, convert_to_tensor=True)
    
    # Сохраняем предложения и их векторы в файл
    with open(output_file, 'wb') as f:
        pickle.dump((sentences, sentence_embeddings), f)
    
    print(f"Векторное пространство сохранено в {output_file}")

# Пример использования
if __name__ == "__main__":
    directory = r'C:\Users\fff\Desktop\BD\JavaNotes'  # Путь к вашей папке
    
    # Создание и сохранение векторного пространства
    create_and_save_vector_space(directory, output_file='vector_space.pkl')
