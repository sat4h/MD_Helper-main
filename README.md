# AI ассистент для локальной wiki .md

Задача сделать чат-бота, ассистента по локальному .md вики файлу.

Было использовано:

- PyCharm Professional 2024.1.4
- LM Studio
- Telegram (Для бота)

В рамках данной работы [пока что] протестирована одна модель **meta-llama-3.1-8b**

Нужно скачать:

```sh
$ pip install python-telegram-bot --upgrade
```

```sh
pip install openai
```

```sh
pip install python-dotenv
```

## vectorization.py

С помощью файла [vectorization.py](vectorization.py) собраются все тексты из .md-файлов, разбиваются на предложения и создается для них векторное представление с помощью модели SentenceTransformer. Результаты сохраняются в файл vector_space.pkl

Требуется скачать и импортировать библиотеки

```sh
import os
import nltk
from sentence_transformers import SentenceTransformer
import pickle
```

Здесь указать путь к папке с .md файлами

```sh
if __name__ == "__main__":
    directory = r'C:\Users\fff\Desktop\BD\JavaNotes'  # Путь к вашей папке
    
    # Создание и сохранение векторного пространства
    create_and_save_vector_space(directory, output_file='vector_space.pkl')
```

## Tgbot_main.py

Здесь [Tgbot_main.py](Tgbot_main.py) уже реализован тг-бот. Для него требуется API и URL для работы с нейронкой. 
BASE_URL дефолтное, API_KEY зависит от модели

```sh
BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("API_KEY", "meta-llama-3.1-8b-instruct")
```

Так же нужно взять API TELEGRAM_TOKEN у [BotFather](https://t.me/BotFather). /newbot -> Создать бота, указать имя и получить API.

```sh
TELEGRAM_TOKEN = "API HERE"
```

## Пример использования

![image](https://github.com/user-attachments/assets/050625a8-0e5b-4d6f-903e-a568ae4bfb5f)

![image](https://github.com/user-attachments/assets/4801f256-1778-4aeb-ab64-5bc86a9992b1)

