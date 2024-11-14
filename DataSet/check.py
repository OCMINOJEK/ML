
with open(file="players_data.csv", mode='r', encoding='utf8') as file:
    sett = set()
    p = 0
    g = 0
    a = 0
    m = 0
    for line in file:
        line = line.split(',')
        sett.add(line[4])
        if line[4] == 'Претендент':
            p += 1
        elif line[4] == 'Грандмастер':
            g += 1
        elif line[4] == 'Алмаз I':
            a += 1
        elif line[4] == 'Мастер':
            m += 1
    print(sett, p, g, a, m)