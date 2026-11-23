import math
import re
from collections import Counter
import lemmatizer_adapter as  la

# Набор констант

# КОНСТАНТЫ

MAX_LEN = 300

# Сокращения для защиты точек
ABBR_ALWAYS = (r"(?:(?i:т\.\s?[еднпкч]|ул|пер|пр|просп|д|кв|корп|стр|им|ген|проф|доц|"
               r"акад|канд|докт|экон|физ|мат|тел|см|рис|табл|обл)|[А-ЯЁA-Z])")
ABBR_UNITS = r"(?:гг?|вв?|руб|коп|долл|тыс|млн|млрд|трлн)"

# Стоп-слова
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне
было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб
без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над
больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда
лучше чуть том нельзя такой им более всегда конечно всю между это также которые который которая которое""".split())

# Дополнительные стоп-слова для лемм
LEMMA_STOP = {"весь", "быть", "этот", "тот", "который", "свой", "мочь", "сам",
              "такой", "наш", "ваш", "каждый", "другой", "очень"}

# Единицы измерения и юр. формы
UNITS = {"тыс", "млн", "млрд", "трлн", "руб", "долл", "коп", "км", "кг"}
LEGAL = {"ооо", "зао", "оао", "пао", "ао", "ип"}

STOP_ALL = STOP | LEMMA_STOP | UNITS | LEGAL

# Теги для пропуска (НЕ включаем ADV - наречия важны!)
SKIP_TAGS = {"PR", "CONJ", "NI", "INTJ"}

# Regex для токенизации
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[А-Яа-яЁёA-Za-z]+")

# Regex для вводных слов в начале предложения
INTRO = re.compile(
    r"^(?:Однако|Кроме того|Таким образом|Вместе с тем|Тем не менее|"
    r"Впрочем|Между тем|Также|Как сообщает [^,]{3,40}|По данным [^,]{3,40}|"
    r"По словам [^,]{3,40}),\s*",
    re.I
)

# Приеобразовываем текст в массив предложений
def split_sentences(text):

    text = re.sub(r"\s+", " ", text.strip())

    # Защищаем точки в сокращениях
    text = re.sub(r"\b(" + ABBR_ALWAYS + r")\.", lambda m: m.group(1) + "\0", text)

    # Защищаем единицы измерения (только если дальше строчная/цифра/запятая)
    text = re.sub(r"\b(" + ABBR_UNITS + r")\.(?=\s*[a-zа-яё0-9,;)])",
                  lambda m: m.group(1) + "\0", text)

    # Разделяем по концам предложений
    parts = re.split(r"(?<=[.!?…])\s+(?=[«\"“(\[]?[А-ЯЁA-Z0-9])", text)

    return [p.replace("\0", ".").strip() for p in parts if p.strip()]
#%%
# Разбираем предложения на слова для преобразовани слов в начальную форму используем лемматизатор

# После того как мы мы составили список где сделали разделение предложений на токены и типы, убрали знаки препинания и
# определили числа и сокращения с специальными сокращениями, теперь выделяем только слова, убираем стоп слова это слова которые будут менать алгоритмам TextRank и TF-IDF строить связи

def tokens(sentence):
    res = []
    for raw in TOKEN_RE.findall(la.normalize_yo(sentence)):
        if raw[0].isdigit():                          # числа: 2023, 8.5
            if len(raw) >= 2:
                res.append(raw.replace(",", "."))
            continue
        if raw.isupper() and 2 <= len(raw) <= 6:      # РФ, США, ВВП - не теряем
            w = la.norm(raw)
            if w not in LEGAL:
                res.append(w)
            continue
        w = la.norm(raw)
        if len(w) <= 2 or w in STOP_ALL:              # т, е, г, д из т.е., г. отсеются здесь
            continue
        lemma, tag = la.analyze_word(w)
        if tag in SKIP_TAGS or lemma in STOP_ALL:
            continue
        res.append(lemma)
    return res

# Приводит список к диапазону [0, 1]

def normalize(v):
    m = max(v) if v else 0
    return [x / m for x in v] if m > 0 else [0.0] * len(v)

# Коэффициент Жаккара для двух множеств

def jaccard(a, b):

    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

# PageRank на графе предложений

def textrank(sent_tokens, d=0.85, iters=40):

    n = len(sent_tokens)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    sets = [set(t) for t in sent_tokens]
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and sets[i] and sets[j]:
                c = len(sets[i] & sets[j])
                if c:
                    w[i][j] = c / (math.log(len(sets[i]) + 1) +
                                   math.log(len(sets[j]) + 1) + 1e-9)

    out_sum = [sum(row) or 1.0 for row in w]
    r = [1.0 / n] * n
    for _ in range(iters):
        r = [(1 - d) / n + d * sum(w[j][i] / out_sum[j] * r[j]
                                   for j in range(n))
             for i in range(n)]
    return r


# Информативность каждого предложения по TF-IDF

def tfidf_scores(sent_tokens):

    n = len(sent_tokens)
    df = Counter(t for ts in sent_tokens for t in set(ts))

    scores = []
    for ts in sent_tokens:
        if not ts:
            scores.append(0.0)
            continue
        tf = Counter(ts)
        s = sum(tf[t] * math.log((n + 1) / (df[t] + 0.5)) for t in tf)
        scores.append(s / math.sqrt(len(ts)))
    return scores

# СБОРКА РЕФЕРАТА

# Очищает предложение от мусора, чтобы сэкономить место в реферате
# Удаляет:
# - Короткие скобки: (по данным МВД)
# - Сноски: [1], [23]
# - Вводные слова в начале: Однако, Кроме того, и т.д.
# - Лишние пробелы
# Делает первую букву заглавной

def clean(s):

    # Убираем короткие скобки (до 80 символов внутри)
    s = re.sub(r"\s*\([^)]{0,80}\)", "", s)
    # Убираем сноски [1], [23] и т.п.
    s = re.sub(r"\[\d+\]", "", s)
    # Убираем вводные слова в начале
    s = INTRO.sub("", s)
    # Нормализуем пробелы
    s = re.sub(r"\s+", " ", s).strip()
    # Первая буква заглавная
    if s:
        s = s[:1].upper() + s[1:]
    return s

# Обрезает строку до limit символов, не разрывая слова
# Если limit попадает на середину слова - обрезает до последнего пробела
# Убирает висящие знаки препинания в конце

def cut_at_word(s, limit):

    if len(s) <= limit:
        return s
    s = s[:limit]
    # Если есть пробел - обрезаем до него
    if " " in s:
        s = s[:s.rfind(" ")]
    # Убираем висящие знаки препинания
    return s.rstrip(" ,;:-—")

# Создаем главную функцию, теперь мы ее разделили на тяжелую часть и лугкую

# Тяжёлая часть: считается один раз на текст

def prepare(text, d=0.85):

    text = (text or "").strip()
    sents = [s for s in (clean(x) for x in split_sentences(text)) if s]
    if not sents:
        return {"raw": text, "sents": []}
    toks = [tokens(s) for s in sents]
    n = len(sents)
    df = Counter(t for ts in toks for t in set(ts))
    tf = Counter(t for ts in toks for t in ts)
    tfidf = normalize([
        sum(tf[t] * math.log((n + 1) / (df[t] + 0.5)) for t in set(ts)) / (len(ts) ** 0.5 + 1e-9)
        if ts else 0.0 for ts in toks])
    return {"raw": text, "sents": sents, "sets": [set(t) for t in toks],
            "tr": normalize(textrank(toks, d=d)), "tfidf": tfidf, "lens": [len(s) for s in sents]}

# Легкая часть

# Собирает реферат из подготовленного текста

# Аргументы:
#     f - результат prepare(text)
#     max_len - максимальная длина реферата
#     w_tr - вес TextRank
#     w_tf - вес TF-IDF
#     w_pos - вес позиции
#     mmr - штраф за повторы

def select(f, max_len=300, w_tr=0.45, w_tf=0.35, w_pos=0.20, mmr=0.5):

    sents = f["sents"]

    if not sents:
        return f["raw"][:max_len]
    if len(sents) == 1:
        return cut_at_word(sents[0], max_len)

    n = len(sents)
    sets = f["sets"]
    lens = f["lens"]

    # Позиционный бонус
    pos = normalize([1.0 / math.sqrt(i + 1) for i in range(n)])

    # Итоговый score
    score = []
    for i in range(n):
        # Штраф за длину
        if lens[i] < 25:
            pen = 0.5
        elif lens[i] > max_len:
            pen = 0.7
        else:
            pen = 1.0

        s = (w_tr * f["tr"][i] + w_tf * f["tfidf"][i] + w_pos * pos[i]) * pen
        score.append(s)

    # Сортировка от лучших к худшим
    order = sorted(range(n), key=lambda i: -score[i])

    # Жадный выбор с MMR-штрафом
    chosen = []
    total = 0
    cand = set(range(n))

    while cand:
        fit = [i for i in cand if total + lens[i] + (1 if chosen else 0) <= max_len]
        if not fit:
            break

        def adjusted(i):
            if chosen:
                max_sim = max(jaccard(sets[i], sets[j]) for j in chosen)
                return score[i] - mmr * max_sim
            return score[i]

        i = max(fit, key=adjusted)
        total += lens[i] + (1 if chosen else 0)
        chosen.append(i)
        cand.remove(i)

    if not chosen:
        return cut_at_word(sents[order[0]], max_len)

    # Сборка в исходном порядке
    parts = {i: sents[i] for i in chosen}
    return " ".join(parts[i] for i in sorted(parts))[:max_len]

# Главная функция: готовит текст и собирает реферат

def summarize(text, max_len=300, w_tr=0.45, w_tf=0.35, w_pos=0.20, mmr=0.5):
    f = prepare(text)
    return select(f, max_len=max_len, w_tr=w_tr, w_tf=w_tf, w_pos=w_pos, mmr=mmr)
