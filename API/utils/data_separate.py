import pandas as pd

data1 = pd.read_csv("paired_data_v0.0.1.csv")
data2 = pd.read_csv("paired_data_v1.0.0.csv")


data1_id = data1["id"]

checker = [0] * 100000
for i in range(len(data1)):
    checker[data1.iloc[i]["id"]] = 1

store = []
for i in range(len(data2)):
    id = data2.iloc[i]["id"]
    if not checker[id]:
        store.append(i)


reward_model_data_ids = store[:600]
lr_model_data_ids = store[600:]

reward_model_data = data2.iloc[reward_model_data_ids]
lr_model_data = data2.iloc[lr_model_data_ids]

reward_model_data.to_csv("reward_model_data_v1.0.0.csv", index=False)
lr_model_data.to_csv("lr_model_data_v1.0.0.csv", index=False)



