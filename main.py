import requests
from bs4 import BeautifulSoup
import csv

url_main = "https://www.leagueofgraphs.com"
url_rating = "/ru/rankings/summoners"
url_ratings_page_n = "/ru/rankings/summoners/page-"
url_champions = "/ru/champions/builds"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}


def get_soup(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url_main + url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None


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
        else:
            continue
    return listt


def get_list_favorites_champions(url: str) -> list:
    response = requests.get(url_main + url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find("table", class_="data_table sortable_table").find_all("tr", class_="")
    listt = []
    for row in rows:
        champ_column = row.find("td", class_="champColumn")
        if champ_column:
            listt.append(champ_column.find("div", class_="name").get_text().strip())
        else:
            continue
    return listt

def save_to_tsv(data, filename="players_data.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        # Запись заголовков
        writer.writerow(["Rating", "Name", "Games Played", "Win Rate", "Rank", "LP", "Average Enemy Rating", "Kills", "Deaths", "Assists", "Favorite Role", "Favorite Champions"])
        # Запись данных
        for row in data:
            writer.writerow(row)

def parsing_page(url: str):
    rows = get_soup(url).find_all('tr')
    for row in rows:
        # рейтинг игрока
        td_rating = row.find('td', class_='text-right hide-for-small-down')
        # имя игрока
        span_name = row.find('span', class_='name')
        # ссылка на игрока
        td_url_player = row.find('td', class_=None)
        player_link = td_url_player.find('a', href=True) if td_url_player else None
        url_player = player_link['href'] if player_link else None
        if not td_rating or not span_name or not url_player:
            continue
        rating = td_rating.get_text(strip=True)[:-1]
        name = span_name.get_text(strip=True)
        # страница игрока
        response_player = get_soup(url_player)
        if response_player is None:
            data.append([rating, name] + [None for i in range(10)])
            continue
        # кол-во сыгранных матчей
        played_matches = response_player.find('div', id='graphDD4', class_='pie-chart small').get_text(strip=True)
        # винрейт
        winrate = response_player.find('div', id='graphDD5', class_='pie-chart small').get_text(strip=True)
        rank_and_LP = response_player.find('div', class_="leagueTier").get_text().split('\n')
        # Звание
        rank = rank_and_LP[1].strip()
        # LP
        LP = rank_and_LP[2].strip().split(' ')[0]
        # средний рейтинг врагов
        sr_rank_enemy = response_player.find('div', class_="leagueTier no-margin-bottom").get_text(strip=True)
        # Список любимых чемпионов
        list_favorites_champions = get_list_favorites_champions(url_player)
        # KDA
        tabs_content_divs = response_player.find_all('div', class_='tabs-content')
        Kills = None
        Deaths = None
        Assists = None
        favorite_role = None

        for tabs_content_div in tabs_content_divs:
            champions_data_div = tabs_content_div.find('div', class_='content', attrs={'data-tab-id': 'championsData-all'})
            if Kills is not None and Deaths is not None and Assists is not None and favorite_role is not None:
                break
            if champions_data_div and champions_data_div.find('div',
                                                              class_='number') and Kills is None and Deaths is None and Assists is None:
                KDA = champions_data_div.find('div', class_='number')
                Kills = KDA.find('span', class_="kills").get_text(strip=True)
                Deaths = KDA.find('span', class_="deaths").get_text(strip=True)
                Assists = KDA.find('span', class_="assists").get_text(strip=True)
            if tabs_content_div and favorite_role is None:
                table_rows_role = tabs_content_div.find_all('th', class_='sortable_column text-left-dark-only')
                for row_role in table_rows_role:
                    if row_role.get_text(strip=True) == "Роль":
                        favorite_role = tabs_content_div.find('div', class_="txt name").get_text(strip=True)
                        break
        data.append(
            [rating, name, played_matches, winrate, rank, LP, sr_rank_enemy, Kills, Deaths, Assists, favorite_role,
             ', '.join(list_favorites_champions)])
        print(
            f"Рейтинг: {rating}, Имя: {name}, Сыграно: {played_matches}, Винрейт: {winrate}, Звание: {rank}, LP {LP}, Среднее звание врагов: {sr_rank_enemy}, Kills: {Kills} Deaths: {Deaths}, Assists: {Assists}, Любимая роль: {favorite_role}, " + "Любимые персонажи: " + ' '.join(
                list_favorites_champions))


data = []
parsing_page(url_rating)
for i in range(2,11):
    parsing_page(url_ratings_page_n + str(i))
save_to_tsv(data)
