# ИМПОРТЫ

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Считает количество предложений в тексте

def count_sentences(text):

    if not text:
        return 0

    # Защищаем сокращения (упрощённо)
    text = re.sub(
        r"(?:т\.\s?[еднпкч]|ул|пер|д|кв|г|гг|млн|млрд|руб|долл|тыс)\.",
        "X",
        text,
        flags=re.I
    )

    # Считаем знаки конца предложений
    return len(re.findall(r"[.!?…]+", text))

# Считает количество слов в тексте

def count_words(text):

    if not text:
        return 0
    return len(re.findall(r"[а-яА-ЯёЁa-zA-Z]+", text))

# Загружает тексты

def load_dataset(texts_path, refs_path=None):

    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    refs = None
    if refs_path and os.path.exists(refs_path):
        with open(refs_path, "r", encoding="utf-8") as f:
            refs = json.load(f)

    return texts, refs

# Выводит описательную статистику для массива чисел

def print_stats(values, name, unit=""):

    arr = np.array(values)

    print(f"\n    {name}:")
    print(f"      Минимум:    {arr.min():.1f} {unit}")
    print(f"      Q1 (25%):   {np.percentile(arr, 25):.1f} {unit}")
    print(f"      Медиана:    {np.median(arr):.1f} {unit}")
    print(f"      Среднее:    {arr.mean():.1f} {unit}")
    print(f"      Q3 (75%):   {np.percentile(arr, 75):.1f} {unit}")
    print(f"      Максимум:   {arr.max():.1f} {unit}")
    print(f"      Стд.откл.:  {arr.std():.1f} {unit}")


# Проводит полный первичный анализ датасета

def analyze_dataset(texts_path, refs_path=None, dataset_name=None, save_plots=True):

    # Определяем название датасета
    if dataset_name is None:
        dataset_name = os.path.basename(texts_path).replace(".json", "")

    print("=" * 80)
    print(f" ПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА: {dataset_name}")
    print("=" * 80)

    # 1. Загрузка данных

    print("\n Загрузка данных...")
    texts, refs = load_dataset(texts_path, refs_path)

    file_size = os.path.getsize(texts_path) / (1024 * 1024)
    print(f"   Размер файла: {file_size:.2f} МБ")
    print(f"   Количество текстов: {len(texts)}")

    if refs:
        print(f"   Количество эталонов: {len(refs)}")
        if len(texts) != len(refs):
            print(f"   ️ ВНИМАНИЕ: количество текстов и эталонов не совпадает!")

    # 2. Вычисление метрик для каждого текста

    text_lengths_chars = [len(t) for t in texts]
    text_lengths_words = [count_words(t) for t in texts]
    text_sentences_count = [count_sentences(t) for t in texts]

    # Средняя длина предложения (слов)
    avg_sent_lengths = []
    for t in texts:
        n_sents = count_sentences(t)
        n_words = count_words(t)
        if n_sents > 0:
            avg_sent_lengths.append(n_words / n_sents)
        else:
            avg_sent_lengths.append(0)

    # 3. Вывод статистики по текстам

    print("\n" + "=" * 80)
    print(" СТАТИСТИКА ПО ТЕКСТАМ")
    print("=" * 80)

    print_stats(text_lengths_chars, "Длина текста (символы)")
    print_stats(text_lengths_words, "Длина текста (слова)")
    print_stats(text_sentences_count, "Количество предложений", "шт.")
    print_stats(avg_sent_lengths, "Средняя длина предложения", "слов")

    # 4. Статистика по эталонам

    if refs:
        print("\n" + "=" * 80)
        print(" СТАТИСТИКА ПО ЭТАЛОННЫМ РЕФЕРАТАМ")
        print("=" * 80)

        ref_lengths_chars = [len(r) for r in refs]
        ref_lengths_words = [count_words(r) for r in refs]

        print_stats(ref_lengths_chars, "Длина эталона (символы)")
        print_stats(ref_lengths_words, "Длина эталона (слова)")

        # Compression ratio: какую долю от текста занимает реферат
        compression = [len(r) / len(t) if len(t) > 0 else 0
                       for r, t in zip(refs, texts)]
        print_stats([c * 100 for c in compression], "Сжатие (реферат/текст)", "%")

        print(f"\n    Интерпретация:")
        print(f"      Эталон в среднем составляет {np.mean(compression) * 100:.1f}% от длины текста")
        print(f"      Это означает сжатие примерно в {1 / np.mean(compression):.1f} раза")

    # 5. Анализ аномалий

    print("\n" + "=" * 80)
    print(" АНАЛИЗ АНОМАЛИЙ")
    print("=" * 80)

    empty_texts = sum(1 for t in texts if not t.strip())
    very_short = sum(1 for t in texts if len(t) < 50)
    very_long = sum(1 for t in texts if len(t) > 20000)
    no_sentences = sum(1 for t in texts if count_sentences(t) == 0)

    print(f"\n   Пустые тексты:              {empty_texts}")
    print(f"   Очень короткие (<50 симв.): {very_short}")
    print(f"   Очень длинные (>20K симв.): {very_long}")
    print(f"   Без знаков препинания:      {no_sentences}")

    if refs:
        empty_refs = sum(1 for r in refs if not r.strip())
        too_long_refs = sum(1 for r in refs if len(r) > 500)
        print(f"   Пустые эталоны:             {empty_refs}")
        print(f"   Слишком длинные эталоны:    {too_long_refs}")

    # 6. Частотный анализ слов

    print("\n" + "=" * 80)
    print(" ЧАСТОТНЫЙ АНАЛИЗ СЛОВ")
    print("=" * 80)

    # Стоп-слова для частотного анализа
    STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне
    было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
    опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
    чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь
    этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
    хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
    этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это также""".split())

    # Считаем частоты слов (на первых 1000 текстах для скорости)
    word_counter = Counter()
    for t in texts[:1000]:
        words = re.findall(r"[а-яА-ЯёЁ]+", t.lower())
        for w in words:
            if len(w) > 3 and w not in STOP:
                word_counter[w] += 1

    print(f"\n   Топ-20 самых частых слов (по первым 1000 текстам):")
    for i, (word, count) in enumerate(word_counter.most_common(20), 1):
        print(f"      {i:2d}. {word:<20s} ({count} раз)")

    # 7. Построение графиков

    print("\n" + "=" * 80)
    print(" ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 12))

    # График 1: Распределение длин текстов
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(text_lengths_chars, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(text_lengths_chars), color='red', linestyle='--',
                label=f'Медиана: {np.median(text_lengths_chars):.0f}')
    ax1.set_xlabel("Длина текста (символы)", fontsize=11)
    ax1.set_ylabel("Количество", fontsize=11)
    ax1.set_title("Распределение длин текстов", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # График 2: Распределение числа предложений
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(text_sentences_count, bins=50, color='#FF6B6B', edgecolor='black', alpha=0.7)
    ax2.axvline(np.median(text_sentences_count), color='red', linestyle='--',
                label=f'Медиана: {np.median(text_sentences_count):.0f}')
    ax2.set_xlabel("Число предложений", fontsize=11)
    ax2.set_ylabel("Количество", fontsize=11)
    ax2.set_title("Распределение числа предложений", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # График 3: Распределение длин эталонов
    ax3 = fig.add_subplot(2, 3, 3)
    if refs:
        ref_lengths_chars = [len(r) for r in refs]
        ax3.hist(ref_lengths_chars, bins=50, color='#45B7D1', edgecolor='black', alpha=0.7)
        ax3.axvline(np.median(ref_lengths_chars), color='red', linestyle='--',
                    label=f'Медиана: {np.median(ref_lengths_chars):.0f}')
        ax3.axvline(300, color='green', linestyle='--', linewidth=2,
                    label='Лимит 300 симв.')
        ax3.set_xlabel("Длина эталона (символы)", fontsize=11)
        ax3.set_ylabel("Количество", fontsize=11)
        ax3.set_title("Распределение длин эталонов", fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, "Эталоны не загружены", ha='center', va='center',
                 transform=ax3.transAxes, fontsize=14)
        ax3.set_title("Эталоны отсутствуют", fontsize=13)

    # График 4: Box plot (сравнение текстов и эталонов)
    ax4 = fig.add_subplot(2, 3, 4)
    if refs:
        data_to_plot = [text_lengths_chars, ref_lengths_chars]
        labels = ['Тексты', 'Эталоны']
    else:
        data_to_plot = [text_lengths_chars]
        labels = ['Тексты']

    bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = ['#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel("Длина (символы)", fontsize=11)
    ax4.set_title("Box plot: Тексты vs Эталоны", fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # График 5: Степень сжатия
    ax5 = fig.add_subplot(2, 3, 5)
    if refs:
        compression_pct = [c * 100 for c in compression]
        ax5.hist(compression_pct, bins=50, color='#FFA07A', edgecolor='black', alpha=0.7)
        ax5.axvline(np.median(compression_pct), color='red', linestyle='--',
                    label=f'Медиана: {np.median(compression_pct):.1f}%')
        ax5.set_xlabel("Сжатие (реферат/текст), %", fontsize=11)
        ax5.set_ylabel("Количество", fontsize=11)
        ax5.set_title("Степень сжатия текста", fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, "Эталоны не загружены", ha='center', va='center',
                 transform=ax5.transAxes, fontsize=14)
        ax5.set_title("Сжатие недоступно", fontsize=13)

    # График 6: Scatter
    ax6 = fig.add_subplot(2, 3, 6)
    if refs:
        # Берём подвыборку для читаемости
        sample_size = min(1000, len(texts))
        sample_idx = np.random.choice(len(texts), sample_size, replace=False)
        x = [text_lengths_chars[i] for i in sample_idx]
        y = [ref_lengths_chars[i] for i in sample_idx]
        ax6.scatter(x, y, alpha=0.5, c='#45B7D1', edgecolors='black', s=20)
        ax6.plot([0, max(x)], [0, max(x)], 'r--', alpha=0.5, label='y = x')
        ax6.set_xlabel("Длина текста (символы)", fontsize=11)
        ax6.set_ylabel("Длина эталона (символы)", fontsize=11)
        ax6.set_title("Текст vs Эталон", fontsize=13, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Эталоны не загружены", ha='center', va='center',
                 transform=ax6.transAxes, fontsize=14)
        ax6.set_title("Scatter недоступен", fontsize=13)

    plt.suptitle(f"Анализ датасета: {dataset_name}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

    # 8. Итоговая сводка

    print("\n" + "=" * 80)
    print(" ИТОГОВАЯ СВОДКА")
    print("=" * 80)

    print(f"\n   Датасет:                {dataset_name}")
    print(f"   Количество текстов:     {len(texts)}")
    print(f"   Размер файла:           {file_size:.2f} МБ")
    print(
        f"   Средняя длина текста:   {np.mean(text_lengths_chars):.0f} симв. / {np.mean(text_lengths_words):.0f} слов")
    print(f"   Медиана предложений:    {np.median(text_sentences_count):.0f}")

    if refs:
        print(f"   Средняя длина эталона:  {np.mean(ref_lengths_chars):.0f} симв.")
        print(f"   Среднее сжатие:         {np.mean(compression) * 100:.1f}%")

    print("\n" + "=" * 80)

    return {
        "texts": texts,
        "refs": refs,
        "stats": {
            "n_texts": len(texts),
            "avg_text_len": float(np.mean(text_lengths_chars)),
            "median_text_len": float(np.median(text_lengths_chars)),
            "avg_sentences": float(np.mean(text_sentences_count)),
            "median_sentences": float(np.median(text_sentences_count)),
        }
    }


# Сравнивает два датасета и показывает, насколько они статистически похожи

def compare_datasets(path1, path2, name1=None, name2=None):

    if name1 is None:
        name1 = os.path.basename(path1).replace(".json", "")
    if name2 is None:
        name2 = os.path.basename(path2).replace(".json", "")

    print("=" * 80)
    print(f" СРАВНЕНИЕ ДАТАСЕТОВ: {name1} vs {name2}")
    print("=" * 80)

    # Загрузка данных
    texts1, _ = load_dataset(path1)
    texts2, _ = load_dataset(path2)

    # Вычисление метрик
    lens1 = np.array([len(t) for t in texts1])
    lens2 = np.array([len(t) for t in texts2])
    sents1 = np.array([count_sentences(t) for t in texts1])
    sents2 = np.array([count_sentences(t) for t in texts2])

    # Таблица сравнения
    print(f"\n{'Метрика':<30} {name1:<20} {name2:<20}")
    print("-" * 70)
    print(f"{'Количество текстов':<30} {len(texts1):<20} {len(texts2):<20}")
    print(f"{'Средняя длина (симв.)':<30} {lens1.mean():<20.1f} {lens2.mean():<20.1f}")
    print(f"{'Медиана длины (симв.)':<30} {np.median(lens1):<20.1f} {np.median(lens2):<20.1f}")
    print(f"{'Среднее число предложений':<30} {sents1.mean():<20.1f} {sents2.mean():<20.1f}")
    print(f"{'Медиана предложений':<30} {np.median(sents1):<20.1f} {np.median(sents2):<20.1f}")

    # Анализ схожести распределений
    print(f"\n Анализ схожести распределений:")
    mean_diff_pct = abs(lens1.mean() - lens2.mean()) / ((lens1.mean() + lens2.mean()) / 2) * 100
    median_diff_pct = abs(np.median(lens1) - np.median(lens2)) / ((np.median(lens1) + np.median(lens2)) / 2) * 100

    print(f"   Разница средних длин:     {mean_diff_pct:.2f}%")
    print(f"   Разница медиан:           {median_diff_pct:.2f}%")

    if mean_diff_pct < 5:
        print(f"    Распределения очень похожи - выборки репрезентативны")
    elif mean_diff_pct < 15:
        print(f"   ️ Небольшое расхождение - допустимо")
    else:
        print(f"    Значительное расхождение - выборки могут отличаться")

    # Визуализация сравнения
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Длины текстов
    axes[0].hist(lens1, bins=50, alpha=0.5, label=name1, color='#4ECDC4', edgecolor='black')
    axes[0].hist(lens2, bins=50, alpha=0.5, label=name2, color='#FF6B6B', edgecolor='black')
    axes[0].set_xlabel("Длина текста (символы)", fontsize=11)
    axes[0].set_ylabel("Количество", fontsize=11)
    axes[0].set_title("Сравнение длин текстов", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Число предложений
    axes[1].hist(sents1, bins=50, alpha=0.5, label=name1, color='#4ECDC4', edgecolor='black')
    axes[1].hist(sents2, bins=50, alpha=0.5, label=name2, color='#FF6B6B', edgecolor='black')
    axes[1].set_xlabel("Число предложений", fontsize=11)
    axes[1].set_ylabel("Количество", fontsize=11)
    axes[1].set_title("Сравнение числа предложений", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()