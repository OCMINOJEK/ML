def one_hot(rank: str):
    oh_line = ['0','0','0','0']
    if rank == 'Претендент':
        oh_line[0] = '1'
    elif rank == 'Грандмастер':
        oh_line[1] = '1'
    elif rank == 'Мастер':
        oh_line[2] = '1'
    elif rank == 'Алмаз I':
        oh_line[3] = '1'
    elif rank == '?':
        pass
    else:
        raise ValueError('чет не то')
    return oh_line


with open(file="processed_data.csv", mode='r', encoding='utf8') as file:
    dict_rank = {}
    p, g, a, m = 0, 0, 0, 0
    for line in file:
        line = line.split(',')
        if line[2] not in dict_rank:
            dict_rank[line[2]] = 0
        elif line[2] == 'Претендент':
            dict_rank[line[2]] += 1
        elif line[2] == 'Грандмастер':
            dict_rank[line[2]] += 1
        elif line[2] == 'Алмаз I':
            dict_rank[line[2]] += 1
        elif line[2] == 'Мастер':
            dict_rank[line[2]] += 1
    print(dict_rank)
ranks = ["Challenger", "GrandMaster", "Master", "Diamond I"]
with open(file="processed_data.csv", mode='r', encoding='utf-8') as file:
    header = file.readline()
    data = file.readlines()
print(header)
header = header.split(',')
new_header = header[0:2]
new_header += ['is ' + rank for rank in ranks]
new_header += header[3:]
print(new_header)

with open(file="processed_data2.csv", mode='w', encoding='utf-8') as file:
    file.write(','.join(new_header))
    for line in data:
        line = line.strip().split(',')
        new_line = line[0:2]
        new_line += one_hot(line[2])
        new_line += line[3:]
        file.write(','.join(new_line) + '\n')

