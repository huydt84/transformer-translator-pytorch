with open("data/src/jaen/train.ja", "r", encoding="utf-8") as f:
    ja_lst1 = f.readlines()

with open("data/src/javi/train.ja", "r", encoding="utf-8") as f:
    ja_lst2 = f.readlines()

with open("data/src/envi/train.en", "r", encoding="utf-8") as f:
    en_lst = f.readlines()

with open("data/trg/envi/train.vi", "r", encoding="utf-8") as f:
    vi_lst1 = f.readlines()

with open("data/trg/javi/train.vi", "r", encoding="utf-8") as f:
    vi_lst2 = f.readlines()

src = ja_lst1 + ja_lst2 + en_lst + vi_lst1 + vi_lst2
trg = vi_lst1 + vi_lst2 + en_lst

with open("data/raw_data.src", "w", encoding="utf-8") as f:
    for s in src:
        f.write(s.strip())
        f.write("\n")

with open("data/raw_data.trg", "w", encoding="utf-8") as f:
    for s in trg:
        f.write(s.strip())
        f.write("\n")