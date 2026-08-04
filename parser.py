"""
Парсер объявлений о продаже квартир с украинского OLX.

Модуль выполняет три основные задачи:
    1. Сбор ссылок на объявления со страниц каталога.
    2. Скачивание HTML-страниц каждого объявления на диск.
    3. Парсинг HTML и формирование структурированного CSV-датасета.

Выходной датасет содержит 9 признаков, закодированных в числовом виде,
пригодных для дальнейшего анализа или машинного обучения.
"""

from bs4 import BeautifulSoup
import os
import time
import csv
import re
import requests
from urllib.parse import urljoin
from datetime import datetime
from typing import List, Dict, Optional


# =============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# =============================================================================

# Заголовки HTTP-запросов для имитации реального браузера.
# Это помогает обойти базовую защиту от ботов и получить корректный HTML.
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7',
}

# Текущая дата сессии (используется как референсная точка при необходимости).
CURRENT_DATE = datetime(2026, 8, 4)


# =============================================================================
# ЭТАП 1: СБОР ССЫЛОК НА ОБЪЯВЛЕНИЯ
# =============================================================================

def get_apartment_links(catalog_url: str, max_pages: int = 1) -> List[str]:
    """
    Собирает уникальные ссылки на отдельные объявления со страниц каталога OLX.

    Алгоритм:
        1. Последовательно обходит страницы каталога (page=1, page=2, ...).
        2. На каждой странице ищет все теги <a> с атрибутом href.
        3. Фильтрует только те ссылки, которые содержат маркеры объявлений.
        4. Преобразует относительные URL в абсолютные через urljoin().
        5. Устраняет дубликаты, сохраняя порядок первого появления.

    Args:
        catalog_url: Базовый URL каталога (например, страница продажи квартир).
        max_pages: Сколько страниц каталога обойти. По умолчанию 1.

    Returns:
        Список уникальных абсолютных URL объявлений.
    """
    apartment_links: List[str] = []

    for page in range(1, max_pages + 1):
        # Формируем URL страницы: для первой страницы — без ?page,
        # для остальных — добавляем параметр пагинации.
        url = f"{catalog_url}?page={page}" if page > 1 else catalog_url
        print(f"[+] Скачиваем страницу каталога #{page}: {url}")

        try:
            # Выполняем GET-запрос с таймаутом 10 секунд
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()  # Вызовет исключение при HTTP-ошибке (4xx, 5xx)
        except requests.RequestException as e:
            print(f"[-] Ошибка при запросе каталога: {e}")
            continue

        # Парсим HTML страницы каталога
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)

        # Фильтруем ссылки: ищем маркеры пути, характерные для объявлений OLX
        for link in links:
            href = link['href']
            if '/uk/obyavlenie/' in href or '/d/uk/' in href or 'uk/d/' in href:
                # Преобразуем относительный URL в абсолютный
                full_url = urljoin(catalog_url, href)
                if full_url not in apartment_links:
                    apartment_links.append(full_url)

        # Задержка между запросами к каталогу — вежливый скрейпинг,
        # снижает риск блокировки IP.
        time.sleep(1.5)

    print(f"[✓] Найдено уникальных объявлений: {len(apartment_links)}")
    return apartment_links


# =============================================================================
# ЭТАП 2: СКАЧИВАНИЕ HTML ОБЪЯВЛЕНИЙ
# =============================================================================

def fetch_and_save_apartments_html(
    links: List[str],
    output_dir: str = "saved_apartments"
) -> List[str]:
    """
    Скачивает HTML-страницы каждого объявления и сохраняет их на диск.

    Файлы именуются последовательно: apartment_1.html, apartment_2.html и т.д.
    Это позволяет отделить этап сбора данных от этапа парсинга и
    повторно парсить данные без повторных сетевых запросов.

    Args:
        links: Список URL объявлений для скачивания.
        output_dir: Папка для сохранения HTML-файлов. Создаётся, если не существует.

    Returns:
        Список путей к успешно сохранённым файлам.
    """
    # Создаём директорию для хранения HTML, если она ещё не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files: List[str] = []

    for idx, url in enumerate(links, start=1):
        print(f"{idx}/{len(links)}: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()

            # Формируем путь к файлу и сохраняем "сырой" HTML
            file_path = os.path.join(output_dir, f"apartment_{idx}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            saved_files.append(file_path)

        except requests.RequestException as e:
            print(f"[-] Не удалось скачать {url}: {e}")

        # Увеличенная задержка между запросами к отдельным объявлениям
        # (2 секунды), чтобы не перегружать сервер OLX.
        time.sleep(2)

    return saved_files


# =============================================================================
# ЭТАП 3: ПАРСИНГ HTML И ФОРМИРОВАНИЕ ДАТАСЕТА
# =============================================================================

def _extract_price(soup: BeautifulSoup) -> Optional[int]:
    """
    Извлекает цену из HTML-страницы объявления.

    Ищет заголовок <h3>, содержащий цифры и символ гривны (грн или ₴),
    извлекает числовое значение и возвращает его как целое число.

    Args:
        soup: Объект BeautifulSoup с распарсенным HTML.

    Returns:
        Цена в гривнах как int, или None, если цена не найдена.
    """
    # Ищем заголовок, текст которого содержит цифры и символ валюты
    price_tag = soup.find('h3', string=re.compile(r'[\d\s]+\s*(грн|₴)', re.IGNORECASE))
    if price_tag:
        text = price_tag.get_text(strip=True)
        # Извлекаем группу цифр перед символом валюты
        match = re.search(r'([\d\s]+)[.,]?\d*\s*(грн|₴)', text, re.IGNORECASE)
        if match:
            # Убираем все нецифровые символы (пробелы-разделители тысяч)
            digits_only = re.sub(r'\D', '', match.group(1))
            return int(digits_only)
    return None


def parse_apartment_html(file_path: str) -> Optional[Dict[str, object]]:
    """
    Парсит HTML-файл одного объявления и возвращает словарь с 9 признаками.

    Все категориальные признаки преобразуются в числовые коды,
    числовые признаки — в float/int. Если хотя бы один обязательный
    признак отсутствует, объявление отбрасывается (возвращается None).

    Схема кодирования:
        seller_type  : 0=Бізнес, 1=Приватна особа
        object_type  : 0=Новобудова, 1=Вторинний ринок
        layout       : 0=Студія, 1=Роздільна, 2=Смарт-квартира,
                       3=Вільне планування, 4=Пентхаус
        furnished    : 0=Ні, 1=Так
        renovation   : 0=Євроремонт, 1=Косметичний, 2=Після будівельників,
                       3=Житловий стан, 4=Авторський проект

    Args:
        file_path: Путь к HTML-файлу объявления.

    Returns:
        Словарь с извлечёнными данными, или None, если данные неполные.
    """
    # Загружаем и парсим HTML-файл
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Инициализируем словарь данных пустыми строками (маркер "не найдено")
    data: Dict[str, object] = {
        'seller_type': '',
        'object_type': '',
        'total_area': '',
        'kitchen_area': '',
        'rooms': '',
        'layout': '',
        'furnished': '',
        'renovation': '',
    }

    # --- Извлечение цены (отдельно от параметров) ---
    price = _extract_price(soup)
    if price is not None:
        data['price'] = price

    # --- Извлечение параметров из контейнера объявления ---
    # На OLX параметры объявления (площадь, комнаты и т.д.) находятся
    # в специальном контейнере с data-testid="ad-parameters-container"
    params_container = soup.find(attrs={'data-testid': 'ad-parameters-container'})

    if params_container:
        # Перебираем все параграфы внутри контейнера — каждый содержит
        # пару "ключ: значение" или просто текстовое значение.
        for tag in params_container.find_all('p'):
            text = tag.get_text(strip=True)
            if not text:
                continue

            # Разделяем строку на ключ и значение по первому двоеточию
            if ':' in text:
                key, value = text.split(':', 1)
                key = key.strip()
                value = value.strip()
            else:
                key = text
                value = text

            key_lower = key.lower()
            val_lower = value.lower()

            # --- Кодирование категориальных и числовых признаков ---

            # seller_type: 0=Бізнес, 1=Приватна особа
            if 'приватна особа' in key_lower:
                data['seller_type'] = 1
            elif 'бізнес' in key_lower:
                data['seller_type'] = 0

            # object_type: 0=Новобудова, 1=Вторинний ринок
            elif 'вид об' in key_lower or "об'єкта" in key_lower:
                if 'новобуд' in val_lower:
                    data['object_type'] = 0
                elif 'вторинний' in val_lower:
                    data['object_type'] = 1

            # total_area: извлекаем число с плавающей точкой (м²)
            elif 'загальна площа' in key_lower:
                m = re.search(r'[\d\s]+(?:[.,]\d+)?', value)
                if m:
                    # Убираем пробелы-разделители и заменяем запятую на точку
                    data['total_area'] = float(m.group(0).replace(' ', '').replace(',', '.'))

            # kitchen_area: аналогично total_area
            elif 'площа кухні' in key_lower:
                m = re.search(r'[\d\s]+(?:[.,]\d+)?', value)
                if m:
                    data['kitchen_area'] = float(m.group(0).replace(' ', '').replace(',', '.'))

            # rooms: количество комнат (целое число)
            elif 'кількість кімнат' in key_lower:
                m = re.search(r'\d+', value)
                if m:
                    data['rooms'] = int(m.group(0))

            # layout: планировка (5 категорий)
            elif 'планування' in key_lower:
                if 'студія' in val_lower:
                    data['layout'] = 0
                elif 'роздільна' in val_lower:
                    data['layout'] = 1
                elif 'смарт' in val_lower:
                    data['layout'] = 2
                elif 'вільне' in val_lower:
                    data['layout'] = 3
                elif 'пентхаус' in val_lower:
                    data['layout'] = 4

            # furnished: наличие мебели (0=Ні, 1=Так)
            elif 'меблювання' in key_lower:
                if 'так' in val_lower or 'повністю' in val_lower:
                    data['furnished'] = 1
                elif 'ні' in val_lower:
                    data['furnished'] = 0

            # renovation: тип ремонта (5 категорий)
            elif 'ремонт' in key_lower:
                if 'євроремонт' in val_lower:
                    data['renovation'] = 0
                elif 'косметичний' in val_lower:
                    data['renovation'] = 1
                elif 'після будівельників' in val_lower or 'після будівельн' in val_lower:
                    data['renovation'] = 2
                elif 'житловий' in val_lower:
                    data['renovation'] = 3
                elif 'авторський' in val_lower:
                    data['renovation'] = 4

    # --- Валидация: проверяем, что все обязательные поля заполнены ---
    # Обязательные поля — все кроме цены (цена может отсутствовать в некоторых
    # объявлениях, но для ML-модели она обычно является целевой переменной).
    required = [
        'seller_type', 'object_type', 'total_area',
        'kitchen_area', 'rooms', 'layout', 'furnished', 'renovation'
    ]

    if any(data[field] == '' or data[field] is None for field in required):
        return None

    # Если цена не была извлечена, помечаем как None (можно отфильтровать позже)
    if 'price' not in data:
        data['price'] = None

    return data


def build_dataset(
    html_dir: str = "saved_apartments",
    output_csv: str = "apartments_dataset.csv"
) -> None:
    """
    Парсит все HTML-файлы из указанной папки и формирует CSV-датасет.

    Проходит по всем .html файлам в директории, вызывает parse_apartment_html()
    для каждого, фильтрует неполные записи (None) и записывает результат
    в CSV с BOM (utf-8-sig) для корректного открытия в Excel.

    Args:
        html_dir: Папка с сохранёнными HTML-файлами объявлений.
        output_csv: Имя выходного CSV-файла.
    """
    if not os.path.exists(html_dir):
        print(f"[-] Папка {html_dir} не найдена.")
        return

    all_rows: List[Dict[str, object]] = []

    # Получаем список HTML-файлов и сортируем их для предсказуемого порядка
    html_files = sorted([f for f in os.listdir(html_dir) if f.endswith('.html')])

    for fname in html_files:
        fpath = os.path.join(html_dir, fname)
        print(f"[*] Парсим {fname}...")
        row = parse_apartment_html(fpath)
        if row:
            all_rows.append(row)

    if not all_rows:
        print("[-] Нет данных для сохранения.")
        return

    # Определяем порядок колонок в CSV
    fieldnames = [
        'price', 'seller_type', 'object_type', 'total_area',
        'kitchen_area', 'rooms', 'layout', 'furnished', 'renovation'
    ]

    # Записываем CSV с BOM (utf-8-sig), чтобы Excel корректно распознал
    # кириллицу при открытии файла.
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[✓] Датасет сохранен: {output_csv} ({len(all_rows)} записей)")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR = "saved_apartments"
    CSV_FILE = "apartments_dataset.csv"

    # --- ЭТАП 1: Скачивание ---
    # Раскомментируйте этот блок, если нужно заново собрать HTML с OLX.
    # Скрипт обходит 25 страниц каталога, собирает уникальные ссылки
    # и скачивает HTML каждого объявления.

    all_links = []
    for p in range(25):
        CATALOG_URL = (
            "https://www.olx.ua/uk/nedvizhimost/kvartiry/prodazha-kvartir/"
            f"?page={p+1}"
        )
        print("СТРАНИЦА", p + 1)
        links = get_apartment_links(CATALOG_URL, max_pages=1)
        all_links.extend(links)

    # Удаляем дубликаты, сохраняя порядок
    all_links = list(dict.fromkeys(all_links))
    fetch_and_save_apartments_html(all_links, output_dir=OUTPUT_DIR)
    

    # --- ЭТАП 2: Парсинг уже скачанных HTML ---
    # Этот блок выполняется по умолчанию: парсит локальные HTML-файлы
    # и формирует CSV-датасет без сетевых запросов.
    build_dataset(html_dir=OUTPUT_DIR, output_csv=CSV_FILE)