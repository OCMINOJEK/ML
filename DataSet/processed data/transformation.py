import pandas as pd
from sklearn.preprocessing import StandardScaler

def one_hot_encoding(list_all: list[str], list_data: list[str]) -> list[str]:
    list_ans = ['0'] * len(list_all)
    for i in list_data:
        list_ans[list_all.index(i)] = '1'
    return list_ans


with open(file='../players_data.tsv', mode='r', encoding='utf8') as file_r:
    file_r.readline() # skip first line

    lines = []
    f = open("list_champions.txt", mode='r', encoding='utf8')
    list_champions = f.readline().split(',')
    f.close()

    f = open("list_ranks.txt", mode='r', encoding='utf8')
    list_ranks = f.readline().split(',')
    f.close()

    file_data = file_r.readlines()

dict_data = {"game_played": [], "win_rate": [], "level": [], "kills": [], "deaths": [], "assists": []}

for line in file_data:
    if "?" in line:
        continue
    line = line.strip().split('\t')

    dict_data["game_played"].append(float(line[2]))
    dict_data["win_rate"].append(float(line[3]))
    dict_data["level"].append(float(line[5]))
    dict_data["kills"].append(float(line[7]))
    dict_data["deaths"].append(float(line[8]))
    dict_data["assists"].append(float(line[9]))

df = pd.DataFrame(dict_data)
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


index = -1
new_data = []
for line in file_data:
    if "?" in line:
        continue
    index += 1
    line = line.strip().split('\t')

    line[2] = str(df_normalized["game_played"][index])
    line[3] = str(df_normalized["win_rate"][index])
    line[5] = str(df_normalized["level"][index])
    line[7] = str(df_normalized["kills"][index])
    line[8] = str(df_normalized["deaths"][index])
    line[9] = str(df_normalized["assists"][index])


    imut_1 = line[0:4]
    rank = line[4]
    imut_2 = line[5:6]
    enemy = line[6]
    imut_3 = line[7:11]
    favorite_champions = line[11].strip().split(', ')


    line = imut_1 + one_hot_encoding(list_ranks, [rank]) + imut_2 \
        + one_hot_encoding(list_ranks, [enemy]) + imut_3 + one_hot_encoding(list_champions, favorite_champions)

    line.pop(0)
    line.pop(0)

    new_data.append(line)


with open(file='processed_data.csv', mode='w', encoding='utf8') as file:
    f = open("list_champions_en.txt", mode='r', encoding='utf8')
    list_champions_en = f.readline().split(',')
    f.close()
    f =  open("list_ranks_en.txt", mode='r', encoding='utf8')
    list_ranks_en = f.readline().split(',')
    f.close()
    header = "Games Played,Win Rate %, " + \
             ','.join(["is " + rank for rank in list_ranks_en]) + ",Level," + \
             ','.join(["is Enemy " + rank for rank in list_ranks_en]) + \
             ",Kills,Deaths,Assists,Favorite Role," + \
             ','.join(["is " + champion for champion in list_champions_en])

    file.writelines(header+'\n')
    file.writelines('\n'.join([','.join(line) for line in new_data]))

