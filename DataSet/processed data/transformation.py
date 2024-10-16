with open(file='../players_data.tsv', mode='r', encoding='utf8') as file_r:
    print(file_r.readline()) # skip first line

    lines = []
    f = open("list_champions.txt", mode='r', encoding='utf8')
    list_champions = f.readline().split(',')
    f.close()

    f = open("list_ranks.txt", mode='r', encoding='utf8')
    list_ranks = f.readline().split(',')
    f.close()

    max_games_played = float('-inf')
    min_games_played = float('inf')

    max_level = float('-inf')
    min_level = float('inf')

    max_KDA = [float('-inf') for i in range(3)]
    min_KDA = [float('inf') for i in range(3)]

    for line in file_r:
        if "?" in line:
            continue
        line = line.strip().split('\t')
        favorite_champions = line[11].strip().split(', ')
        line.pop(11)  # delete favorite champions
        for _ in range(len(list_champions)):
            line.append(str(0))
        for i in favorite_champions:
            line[list_champions.index(i) + 11] = str(1)

        line[4] = str(list_ranks.index(line[4]) + 1) # label encoding player ranks 1,2,3...

        line[6] = str(list_ranks.index(line[6]) + 1) # label encoding enemy ranks 1,2,3...

        max_games_played = max(max_games_played, int(line[2]))
        min_games_played = min(min_games_played, int(line[2]))

        max_level = max(max_level, int(line[5]))
        min_level = min(min_level, int(line[5]))

        print(max_level, min_level)

        for i in range(3):
            max_KDA[i] = max(max_KDA[i], float(line[7 + i]))
            min_KDA[i] = min(min_KDA[i], float(line[7 + i]))
        lines.append(line)

    for i in range(len(lines)):


        lines[i][2] = str(round(((float(lines[i][2]) - min_games_played)/(max_games_played - min_games_played)), 3)) # normalization games_played
        lines[i][3] = str(round((float(lines[i][3])/100), 3)) # normalization win_rate

        lines[i][5] = str(round(((float(lines[i][5]) - min_level) / (max_level - min_level)), 3))

        lines[i][7] = str(round(((float(lines[i][7]) - min_KDA[0]) / (max_KDA[0] - min_KDA[0])),3))  # normalization Kills
        lines[i][8] = str(round(((float(lines[i][8]) - min_KDA[1]) / (max_KDA[1] - min_KDA[1])),3))  # normalization Deaths
        lines[i][9] = str(round(((float(lines[i][9]) - min_KDA[2]) / (max_KDA[2] - min_KDA[2])),3))  # normalization Assists

        lines[i].pop(0)  # delete number player
        lines[i].pop(0)  # delete nickname


with open(file='processed_data.csv', mode='w', encoding='utf8') as file:
    f = open("list_champions_en.txt", mode='r', encoding='utf8')
    list_champions_en = f.readline().split(',')
    f.close()
    header = "Games Played,Win Rate %,Rank,Level,Average Enemy Rating,Kills,Deaths,Assists,Favorite Role,"
    for i in range(len(list_champions_en)):
        list_champions_en[i] = "is favorite " + str(list_champions_en[i])
    header_favorite_champions = ','.join(list_champions_en)
    header += header_favorite_champions
    file.writelines(header+'\n ')
    file.writelines('\n'.join([','.join(line) for line in lines]))

