
with open(file="players_data.csv", mode='r', encoding='utf8') as file:
    sett = set()
    for line in file:
        line = line.split(',')
        sett.add(line[3])
    print(sett)