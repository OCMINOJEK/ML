import os

import requests
from Tools.scripts.sortperf import list_sort
from bs4 import BeautifulSoup
import csv

# Основные URL и заголовки для запросов
url_main = "https://www.leagueofgraphs.com"
url_rating = "/ru/rankings/summoners"
url_ratings_page_n = "/ru/rankings/summoners/page-"
url_champions = "/ru/champions/builds"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}


# Функция для получения soup-объекта с проверкой статуса запроса
def get_soup(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url_main + url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None


# Функция для получения списка чемпионов
def get_list_champions() -> list:
    response = requests.get(url_main + url_champions, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr")
    listt = []
    for row in rows:
        row = row.find("div", class_="txt")
        if row:
            listt.append(row.find("span", class_="name").get_text().strip())
    return listt


# Функция для получения списка любимых чемпионов игрока
def get_list_favorites_champions(url: str) -> list:
    try:
        response = requests.get(url_main + url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.find("table", class_="data_table sortable_table").find_all("tr", class_="")
        listt = []
        for row in rows:
            champ_column = row.find("td", class_="champColumn")
            if champ_column:
                listt.append(champ_column.find("div", class_="name").get_text().strip())
        return listt
    except Exception as e:
        print(f"Error fetching favorite champions: {e}")
        return []


# Функция для сохранения данных в файл .tsv
def save_to_tsv(data_to_save, filename="players_data.tsv"):
    with open(filename, mode='a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        # Запись заголовков (если файл пустой)
        if file.tell() == 0:
            writer.writerow([
                "Rating", "Name", "Games Played", "Win Rate, %", "Rank", "Level",
                "Average Enemy Rating", "Kills", "Deaths", "Assists", "Favorite Role", "Favorite Champions"
            ])
        # Запись данных
        writer.writerows(data_to_save)

        global data
        data = []

# Функция для сохранения списка игроков в файл
def save_players_to_file(players, filename="players_list.csv"):
    if os.path.exists(filename):
        print(f"Файл {filename} уже существует. Пропуск сохранения.")
        return
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Запись заголовков
        writer.writerow(["Rating", "Name", "URL"])
        # Запись данных
        for player in players:
            writer.writerow([player[0].get_text(strip=True)[:-1], player[1].get_text(strip=True), player[2]])

# Функция для чтения списка игроков из файла
def read_players_from_file(filename="players_list.csv") -> list:
    players = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Пропустить заголовки
        for row in reader:
            players.append([row[0], row[1], row[2]])  # [Рейтинг, Имя, URL]
    return players

# Функция для получения списка игроков на странице
def get_page_list_players(url: str) -> list:
    soup = get_soup(url)
    if soup is None:
        return []

    rows = soup.find_all('tr')
    page_list_players = []

    for row in rows:
        # Рейтинг игрока
        td_rating = row.find('td', class_='text-right hide-for-small-down')
        # Имя игрока
        span_name = row.find('span', class_='name')
        # Ссылка на игрока
        td_url_player = row.find('td', class_=None)
        player_link = td_url_player.find('a', href=True) if td_url_player else None
        url_player = player_link['href'] if player_link else None

        if td_rating and span_name and url_player:
            page_list_players.append([td_rating, span_name, url_player])

    return page_list_players


# Функция для парсинга данных конкретного игрока
def parsing_player(player: list):
    td_rating, span_name, url_player = player

    # Парсинг рейтинга и имени
    rating = td_rating if td_rating else "?"
    name = span_name if span_name else "?"
    # if int(rating) <= 1:
    #     return

    # Получение данных о странице игрока
    response_player = get_soup(url_player)
    if response_player is None:
        data.append([rating, name] + ["?" for _ in range(10)])
        return

    # Парсинг сыгранных матчей
    try:
        played_matches = response_player.find_all(class_="pie-chart small")[0].get_text(strip=True)

    except Exception:
        played_matches = "?"

    # Парсинг винрейта
    try:
        winrate = response_player.find_all(class_='pie-chart small')[1].get_text(strip=True)[:-1]
    except Exception:
        winrate = "?"

    # Парсинг звания
    try:
        rank_ = response_player.find('div', class_="leagueTier").get_text().split('\n')
        rank = rank_[1].strip()
    except Exception:
        rank = "?"

    #Парсинг уровня
    try:
        level = response_player.find('div', class_="bannerSubtitle").get_text().strip()
        level = level[level.find('Уровень') + 7: ].split(' ')[1]
    except Exception:
        level = "?"

    # Средний рейтинг врагов
    try:
        sr_rank_enemy = response_player.find('div', class_="leagueTier no-margin-bottom").get_text(strip=True)
    except Exception:
        sr_rank_enemy = "?"

    # Любимые чемпионы
    list_favorites_champions = get_list_favorites_champions(url_player)

    # Парсинг KDA и любимой роли
    Kills, Deaths, Assists, favorite_role = "?", "?", "?", "?"
    try:
        tabs_content_divs = response_player.find_all('div', class_='tabs-content')

        for tabs_content_div in tabs_content_divs:
            if Kills == "?" or Deaths == "?" or Assists == "?":
                champions_data_div = tabs_content_div.find('div', class_='content',
                                                           attrs={'data-tab-id': 'championsData-all'})
                if champions_data_div and champions_data_div.find('div', class_='number'):
                    KDA = champions_data_div.find('div', class_='number')
                    Kills = KDA.find('span', class_="kills").get_text(strip=True)
                    Deaths = KDA.find('span', class_="deaths").get_text(strip=True)
                    Assists = KDA.find('span', class_="assists").get_text(strip=True)

            if favorite_role == "?":
                table_rows_role = tabs_content_div.find_all('th', class_='sortable_column text-left-dark-only')
                for row_role in table_rows_role:
                    if row_role.get_text(strip=True) == "Роль":
                        favorite_role = tabs_content_div.find('div', class_="txt name").get_text(strip=True)
                        break
    except Exception:
        Kills, Deaths, Assists, favorite_role = "?", "?", "?", "?"

    # Добавление данных в итоговый список
    data.append([
        rating, name, played_matches, winrate, rank, level, sr_rank_enemy,
        Kills, Deaths, Assists, favorite_role, ', '.join(list_favorites_champions)
    ])

    # Логирование результатов
    print(
        f"Рейтинг: {rating}, Имя: {name}, Сыграно: {played_matches}, Винрейт: {winrate}, Звание: {rank}, level {level}, Среднее звание врагов: {sr_rank_enemy}, "
        f"Kills: {Kills}, Deaths: {Deaths}, Assists: {Assists}, Любимая роль: {favorite_role}, Любимые персонажи: {', '.join(list_favorites_champions)}"
    )

    # Сохранение каждые 100 записей
    if int(rating) % 100 == 0:
        save_to_tsv(data)


# Основной код для парсинга и сохранения данных
data = []

# list_players = get_page_list_players(url_rating)
# print(f"Собираем данные со страницы 1")
# for i in range(2, 51):  # Диапазон страниц для парсинга
#     list_players += get_page_list_players(url_ratings_page_n + str(i))
#     print(f"Собираем данные со страницы {i}")
# save_players_to_file(list_players)

list_players_from_file = read_players_from_file()

for player in list_players_from_file:
    parsing_player(player)

