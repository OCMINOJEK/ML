import csv

with open("players_data.csv", "w", newline='', encoding="utf8") as csvfile:
    with open(file='players_data.tsv', mode='r', encoding="utf8") as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        csv_writer = csv.writer(csvfile)
        for line in tsv_reader:
            line[11] = ';'.join(str(line[11]).split(', '))
            csv_writer.writerow(line)
