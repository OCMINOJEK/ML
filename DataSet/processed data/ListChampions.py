import requests
from bs4 import BeautifulSoup

url_main = "https://www.leagueofgraphs.com"
url_champions = "/champions/builds"
url_rank = "/ru/rankings/rank-distribution"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}
#список персонажей
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

def get_list_ranks() -> list:
    response = requests.get(url_main + url_rank, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser").find("table", class_="data_table summoners_distribution_table")
    rows = soup.find_all("tr")
    listt = []
    for row in rows:
        row = row.find("td", class_="nowrap")
        if row:
            listt.append(row.get_text().strip())
    return listt

print(get_list_ranks())

with open(file="list_champions_en.txt", mode="w", encoding="utf-8") as f:
    f.write(','.join(get_list_champions()))
with open(file="list_ranks.txt", mode="w", encoding="utf-8") as f:
    f.write(','.join(get_list_ranks()))
