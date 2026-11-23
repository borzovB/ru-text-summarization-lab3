# Адаптер для использования лемматизатора из лабораторной №1 в рефераторе

# Адаптер для использования лемматизатора из лабораторной №1 в рефераторе
# Оптимальный выбор: level_4 (точность 94.85%, высокая скорость)

import pickle
import os
import re
import sys
import importlib.util
from text_lemmatizer import Lemmatizer

# Путь к модели

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "lab1_model.pkl")

# Кэш для ускорения
_cache = {}

# Глобальная переменная для кэширования модели
_model = None

# Символы для очистки
STRIP_CHARS = ',.?!:;«»"\'()[]…—–-'

# Загружает модель

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Lemmatizer":          # вместо перебора всех .py в папке
            return Lemmatizer
        return super().find_class(module, name)

# Загружает модель лемматизатора

def load_lemmatizer():
    global _model
    if _model is None:
        with open(MODEL_FILE, "rb") as f:
            _model = _Unpickler(f).load()["model"]
    return _model


# Приводит 'ё' к 'е' для единообразия обработки текста

def norm(s):
    return s.lower().replace("ё", "е")

# Слово -> (лемма, тег). Всегда нормализует регистр

# Адаптер для использования лемматизатора из лабораторной №1 в рефераторе
# Оптимальный выбор: level_4 (точность 94.85%, высокая скорость)

import pickle
import os
from text_lemmatizer import Lemmatizer

# Путь к модели
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "lab1_model.pkl")

# Глобальные переменные
_model = None
_cache = {}

# Умная нагрузка

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Lemmatizer":
            return Lemmatizer
        return super().find_class(module, name)


def load_lemmatizer():
    """Загружает модель лемматизатора (один раз)."""
    global _model
    if _model is None:
        print(f"Загрузка модели из {MODEL_FILE}...", end=" ", flush=True)
        with open(MODEL_FILE, "rb") as f:
            _model = _Unpickler(f).load()["model"]
        print("готово!")
    return _model

# Нормализация: нижний регистр + ё на е

def norm(s):

    return s.lower().replace("ё", "е")


# Алиас для совместимости
normalize_yo = norm

# ЛЕММАТИЗАЦИЯ

# Лемматизация одного слова с кэшем

def analyze_word(word):

    w = norm(word)
    r = _cache.get(w)
    if r is None:
        m = load_lemmatizer()
        if w in m.forms:
            # level_4 — оптимальный баланс
            r = m.level_4([w])[0]
        else:
            # Вне словаря: берём грубую основу
            r = (w[:6], "S")
        _cache[w] = r
    return r

# Пакетная лемматизация списка слов

def lemmatize_words(words):

    if not words:
        return []

    model = load_lemmatizer()
    words_norm = [norm(w) for w in words]

    # level_4 - оптимальный баланс точности и скорости
    parsed = model.level_4(words_norm)
    return [(lemma, tag) for lemma, tag in parsed]

# Пакетная лемматизация с кэшем

def lemmatize_words_cached(words):

    result = []
    to_process = []
    indices = []

    for i, w in enumerate(words):
        w_norm = norm(w)
        if w_norm in _cache:
            result.append((w_norm, _cache[w_norm]))
        else:
            to_process.append(w_norm)
            indices.append(i)
            result.append(None)

    if to_process:
        lemmas = lemmatize_words(to_process)
        for idx, (lemma, tag) in zip(indices, lemmas):
            w_norm = norm(words[idx])
            _cache[w_norm] = (lemma, tag)
            result[idx] = (w_norm, (lemma, tag))

    return [r[1] if r else (norm(words[i]), (norm(words[i])[:6], "S"))
            for i, r in enumerate(result)]
