import re

string = "(this is some text) adfas (kjd)"

match = re.sub("\(*([^()]*)\)", "", string)

print(match)

with open("data/dict.dat", "r", encoding="utf-8") as f:
    dic = f.readlines()

print(len(dic))

with open("data/dict.ja", "w", encoding="utf-8") as ja:
    with open("data/dict.en", "w", encoding="utf-8") as en:
        for line in dic:
            line_lst = line.strip().split("/")
            word_ja = line_lst[0].split(" ")[0].strip()
            word_en = re.sub("\(*([^()]*)\)", "", line_lst[1]).strip()
            ja.write(word_ja)
            ja.write("\n")
            en.write(word_en)
            en.write("\n")