from tkinter import *
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data.csv")
kitap_liste = df["başlık"].values.tolist()

root = Tk()
var = StringVar()


l = Listbox(root, width=30, height=15, selectmode=MULTIPLE)
count = 0
for x in kitap_liste:
    count+=1
    l.insert(count, x)
l.pack()

df['tür'] = df['tür'].map(lambda x: x.lower().split(','))
df['yazar'] = df['yazar'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    row['yazar'] = ''.join(row['yazar']).lower()

df['anahtar'] = ""

for index, row in df.iterrows():
    konu = row['konu']
    
    r = Rake()

    r.extract_keywords_from_text(konu)
    anahtar_skor = r.get_word_degrees()
    
    row['anahtar'] = list(anahtar_skor.keys())

df.drop(columns = ['konu'], inplace = True)
df.set_index('başlık', inplace = True)

df['anahtarlar'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'yazar':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['anahtarlar'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'anahtarlar'], inplace = True)

count = CountVectorizer()
count_matrix = count.fit_transform(df['anahtarlar'])

indices = pd.Series(df.index)

cosine_sim = cosine_similarity(count_matrix, count_matrix)

def tavsiye_tek(baslik, cosine_sim = cosine_sim):
    tavsiyeler = []
    idx = indices[indices == baslik].index[0]
    skorlar = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top10 = list(skorlar.iloc[1:11].index)
    for i in top10:
        tavsiyeler.append(list(df.index)[i])
    x=0
    var.set("Seçili 1 kitap baz alınarak yapılan ilk 10 tavsiye: \n")
    for i in tavsiyeler:
        x+=1
        var.set(var.get() + "{}. {}\n".format(x,i))

def tavsiye_coklu(*kitaplar, cosine_sim = cosine_sim):
    kitaplar = list(kitaplar)
    count = len(kitaplar)
    tavsiyeler = []
    idx = indices[indices == kitaplar[0]].index[0]
    skor = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    sh = skor.iloc[1:,]
    kitaplar.remove(kitaplar[0])
    x = 0
    for i in kitaplar:
        idx = indices[indices == i].index[0]
        skor = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        shortlist = skor.iloc[1:,]
        if x == 0:
            concat = shortlist.combine(sh, func=(lambda x1, x2: x1 + x2)).sort_values(ascending = False)
        else:
            concat = concat.combine(shortlist, func=(lambda x1, x2: x1 + x2)).sort_values(ascending = False)
    concat = concat.apply(lambda x: x/count)
    top10 = list(concat.iloc[1:11].index)
    for i in top10:
        tavsiyeler.append(list(df.index)[i])
    x=0
    var.set("Seçili {} kitap baz alınarak yapılan ilk 10 tavsiye: \n".format(count))
    for i in tavsiyeler:
        x+=1
        var.set(var.get() + "{}. {}\n".format(x,i))

def select():
    c_list = []
    selected = l.curselection()
    if selected:
        for i in selected:
            c_list.append(kitap_liste[i])
        yolla = tuple(c_list)
        if len(yolla) == 1:
            tavsiye_tek(*yolla)
        else:
            tavsiye_coklu(*yolla)
    else:
        var.set("Kitap seçmediniz.")

calc = Button(root, text="OK", command = select)
calc.pack()
xy = Label(root, textvariable = var)
xy.pack()

root.title("Kitapçı")
root.geometry("300x600")
root.mainloop()

select()
