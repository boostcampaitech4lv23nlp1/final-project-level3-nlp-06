import pandas as pd
import numpy as np

## All hate data
apeach = pd.read_csv("./data/apeach_train.csv")
beep = pd.read_csv("./data/beep_train.csv")
kmhas = pd.read_csv("./data/k_mhas_train.csv")
kold = pd.read_csv("./data/KOLD_train.csv")
unsmile = pd.read_csv("./data/unsmile_train.csv")

store = []
ids = []

count = -1

## Apeach data 정제
## class : 0 = not hate, 1 = hate
## text : ["text"]
print(apeach.keys())
for i in range(len(apeach)):
    now_data = apeach.iloc[i]
    label = now_data["class"]
    text = now_data["text"]
    if label:
        count += 1
        ids.append(count)
        store.append(text)
print(len(store))

## Beep data 정제
## contain_gender_bias = False / True, bias = None / others / gender
## hate = None / offensive / hate
## text : ["comments"]
print(beep.keys())
for i in range(len(beep)):
    now_data = beep.iloc[i]
    label1 = now_data["contain_gender_bias"]
    label2 = now_data["bias"]
    label3 = now_data["hate"]
    text = now_data["comments"]
    if label1 or label2 in ["others", "gender"] or label3 in ["offensive", "hate"]:
        count += 1
        ids.append(count)
        store.append(text)
print(len(store))
    
## K_mhas data 정제
## hate = [1, 2, 3, 4, 5, 6, 7], not hate = [8], ["class"]
## text : ["text"]
print(kmhas.keys())
for i in range(len(kmhas)):
    now_data = kmhas.iloc[i]
    label = eval(now_data["label"])
    text = now_data["text"]
    if 8 not in label:
        count += 1
        ids.append(count)
        store.append(text)
print(len(store))

## kold data 정제
## OFF label로 판단하는 것
## hate : True, not hate : False
## text : ["comment"]
print(kold.keys())
for i in range(len(kold)):
    now_data = kold.iloc[i]
    label = now_data["OFF"]
    text = now_data["comment"]
    if label:
        count += 1
        ids.append(count)
        store.append(text)
print(len(store))

## unsmile data 정제
## if clean = 1 not hate, others is hate 
## text : 
print(unsmile.keys())
for i in range(len(unsmile)):
    now_data = unsmile.iloc[i]
    label = now_data["clean"]
    text = now_data["문장"]
    if not label:
        count += 1
        ids.append(count)
        store.append(text)
print(len(store))

data_dict = {"id": ids, "text": store}
dataset = pd.DataFrame(data_dict)

dataset.to_csv("all_hate.csv", index=False)

