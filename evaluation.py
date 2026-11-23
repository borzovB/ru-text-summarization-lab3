# ОЦЕНКА ROUGE

import itertools
import re
from collections import Counter
import summarizer as s

# Приводим текст к набору токенов, чтобы не ыбло разночтения, вводим однозначность е и ё и убираем все лишнее
# оставляем буквы и числа

def rouge_tokens(text):
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text)

# Главная оценка точности по полноте и точности

def _f1(overlap, n_pred, n_ref):
    if not overlap:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)

# Точность по словам, меняем число совпавших слов в одом сравнении

def rouge_n(pred, ref, n=1):
    p, r = rouge_tokens(pred), rouge_tokens(ref)
    P = Counter(zip(*[p[i:] for i in range(n)]))
    R = Counter(zip(*[r[i:] for i in range(n)]))
    return _f1(sum((P & R).values()), sum(P.values()), sum(R.values()))

# Длина последовательности

def rouge_l(pred, ref):
    p, r = rouge_tokens(pred), rouge_tokens(ref)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[-1]))
        prev = cur
    return _f1(prev[-1], len(p), len(r))

# Метод оценки

def evaluate(preds, refs, max_len):
    res = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        p = p[:max_len]
        res["rouge1"] += rouge_n(p, r, 1)
        res["rouge2"] += rouge_n(p, r, 2)
        res["rougeL"] += rouge_l(p, r)
    return {k: round(v / len(preds), 4) for k, v in res.items()}

# Подбор гиперпараметров

# Подбор оптимальных гиперпараметров на dev-наборе.
#
# Алгоритм:
# 1. Готовим все тексты ОДИН раз (тяжёлая часть: лемматизация, TextRank, TF-IDF)
# 2. Перебираем все комбинации весов (лёгкая часть: только арифметика)
# 3. Для каждой комбинации считаем средний ROUGE
# 4. Возвращаем комбинацию с максимальным ROUGE

def tune(texts, refs, grid, max_len=300, verbose=True):

    if verbose:
        print(f" Подготовка {len(texts)} текстов (тяжёлая часть)...")

    # Готовим все тексты ОДИН раз
    prepared = [s.prepare(t) for t in texts]

    if verbose:
        print(" Подготовка завершена. Перебор комбинаций...")

    # Всего комбинаций: 3 x 3 x 3 x 3 = 81
    total_combos = 81
    if verbose:
        print(f"   Всего комбинаций: {total_combos}")

    best_params = None
    best_score = -1
    best_rouge = None
    tested = 0

    keys = list(grid.keys())
    values = list(grid.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["max_len"] = max_len

        # Генерируем рефераты (лёгкая часть!)
        preds = [s.select(f, **params) for f in prepared]

        # Считаем ROUGE
        scores = evaluate(preds, refs, max_len)
        avg_score = (scores["rouge1"] + scores["rouge2"]) / 2

        tested += 1
        if verbose and tested % 20 == 0:
            print(f"   Проверено {tested}/{total_combos}. Лучший: {best_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            best_rouge = scores.copy()

    if verbose:
        print(f" Перебор завершён. Проверено {tested} комбинаций.")
        print(f" Лучшие параметры: {best_params}")
        print(f" ROUGE: R1={best_rouge['rouge1']:.4f}, "
              f"R2={best_rouge['rouge2']:.4f}, RL={best_rouge['rougeL']:.4f}")

    return best_params, best_score, best_rouge