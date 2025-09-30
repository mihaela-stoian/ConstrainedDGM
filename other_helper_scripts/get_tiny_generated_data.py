import pickle
import pandas as pd
import torch


path = "CTGAN_out/lcld/unconstrained/unconstrained_2_20_500_0.0002_0.0002_22-08-23--16-50-41"
X_train_tiny = pd.read_csv("data/botnet/tiny/train_data.csv")
print(X_train_tiny.shape)
X_val_tiny = pd.read_csv("data/botnet/tiny/val_data.csv")
X_test_tiny = pd.read_csv("data/botnet/tiny/test_data.csv")


with open(f"{path}/unconstrained_generated_data.pkl","rb") as f:
    gen_data = pickle.load(f)
print(gen_data["train"][0].shape)


generated_data = [[], [], []]
sizes = [X_train_tiny.shape[0], X_val_tiny.shape[0], X_test_tiny.shape[0]]
names = ["train", "val", "test"]
for i in range(5):
    for j in range(len(sizes)):
        data_full = gen_data[names[j]][i]

        indices = torch.randperm(len(data_full))[:sizes[j]]
        #data_tiny = data_full.sample(sizes[j].shape[0],random_state=1)
        generated_data[j].append(data_full[indices])


generated_data_tiny = {"train":generated_data[0], "val":generated_data[1], "test":generated_data[2]}
print(generated_data_tiny["train"][0].shape)

with open(f'{path}/unconstrained_generated_data_tiny.pkl', 'wb') as f:
    pickle.dump(generated_data_tiny, f)