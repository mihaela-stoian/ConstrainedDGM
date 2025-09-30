
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from constraints.url import evaluate_numpy_url
from constraints.botnet import evaluate_numpy_botnet
from constraints.lcld import evaluate_numpy_lcld
from constraints.wids import evaluate_numpy_wids
from constraints.news import evaluate_numpy_news
from constraints.faults import evaluate_numpy_faults
from constraints.heloc import evaluate_numpy_heloc, INFINITY

def cons_sat(data, use_case):
    tol=1e-2
    if use_case=="url":
        cons = evaluate_numpy_url(data)

    if use_case=="botnet":
        cons = evaluate_numpy_botnet(data)

    if use_case=="lcld":
        cons = evaluate_numpy_lcld(data)

    if use_case=="wids":
        cons = evaluate_numpy_wids(data)

    if use_case=="heloc":
        cons = evaluate_numpy_heloc(data)

    if use_case=="news":
        cons = evaluate_numpy_news(data)

    if use_case=="faults":
        cons = evaluate_numpy_faults(data)

    count = np.count_nonzero((cons<=tol), axis=1)
    idx_sat = np.argwhere(count==cons.shape[1])
    idx_not_sat = np.argwhere(count!=cons.shape[1])

    return idx_sat, idx_not_sat


def visualize_embeddings(data , labels, label_names, color, graph_path, rs, tsne_comp, pca_comp, perplexity):
    #dim_reducer = TSNE(n_components=2, random_state=281, perplexity=30)
    #pca_30 = PCA(n_components=30, random_state=281)
    dim_reducer = TSNE(n_components=tsne_comp, random_state=rs, perplexity=perplexity)
    pca_30 = PCA(n_components=pca_comp, random_state=rs)
    pca_result_30 = pca_30.fit_transform(data)
    print("PCA completed")
    tsne_results = dim_reducer.fit_transform(pca_result_30)
    df = pd.DataFrame.from_dict({'x':tsne_results[:,0], 'y':tsne_results[:,1], 'label':labels})
    fig, ax = plt.subplots()
    #palette = sns.color_palette("bright", color)
    palette = ["red", "green", "blue", "orange"]
    sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax, palette=palette[:color], alpha=0.4)
    ax.legend(handles=ax.legend_.legendHandles, labels=label_names, loc="lower left")
    plt.tight_layout()
    plt.savefig(graph_path)

def load_pickle(path):
    with  open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def correct_format(data):
    if type(data) == list:
        data = {"train": data[0], "val": data[1], "test": data[2]}
    return data
if __name__ == "__main__":
    use_case="heloc"
    model="TableGAN"
    ordering="random"
    split="train"
    #uncons = load_pickle("TableGAN_Wids_Mihaela/unconstrained_generated_data.pkl")
    uncons = load_pickle("outputs/TableGAN_out/heloc/unconstrained/unconstrained_random_2_adam_200_128_100_64_12-09-23--13-16-16/unconstrained_generated_data.pkl")
    uncons = correct_format(uncons)[split][0]
    #cons = load_pickle(f"TableGAN_Wids_Mihaela/{ordering}_generated_data.pkl")
    cons = load_pickle("outputs/TableGAN_out/heloc/constrained/constrained_random_2_adam_200_128_100_64_12-09-23--21-54-08/generated_data.pkl")
    cons = correct_format(cons)[split][0]
    real = pd.read_csv(f"data/{use_case}/{split}_data.csv")

    color=2
    rs=5
    tsne_comp=2
    pca_comp=10
    perplexity=70

    uncons_labels = np.full((uncons.shape[0],),0)
    cons_labels = np.full((cons.shape[0],),1)
    real_labels = np.full((real.shape[0],),2)

    # uncons_real_data = np.concatenate((uncons, real.to_numpy()))
    # uncons_real_labels = np.concatenate((uncons_labels, real_labels))
    # uncons_real_label_names = ['Unconstrained', 'Real']

    # graph_path = f'plots/{model}_{ordering}_{use_case}_{split}_embeddings_{rs}_{tsne_comp}_{pca_comp}_{perplexity}_uncons_real.png'
    # visualize_embeddings(uncons_real_data, uncons_real_labels, uncons_real_label_names, color, graph_path,  rs, tsne_comp, pca_comp, perplexity)

    # cons_real_data = np.concatenate((cons, real.to_numpy()))
    # cons_real_labels = np.concatenate((cons_labels, real_labels))
    # cons_real_label_names = ['Constrained', 'Real']

    # graph_path = f'plots/{model}_{ordering}_{use_case}_{split}_embeddings_{rs}_{tsne_comp}_{pca_comp}_{perplexity}_cons_real.png'
    # visualize_embeddings(cons_real_data, cons_real_labels, cons_real_label_names, color, graph_path,  rs, tsne_comp, pca_comp, perplexity)

    idx_sat, idx_not_sat = cons_sat(uncons, use_case)
    uncons_cons_data = np.concatenate((uncons, cons))
    uncons_cons_labels = np.full(uncons.shape[0]+cons.shape[0], 2)
    uncons_cons_labels[idx_not_sat]=0
    uncons_cons_labels[idx_sat]=1
    uncons_cons_label_names = [ "TableGAN cons violated", 'TableGAN cons satisfied', 'C-TableGAN']
    graph_path = f'plots/{model}_{ordering}_{use_case}_{split}_embeddings_{rs}_{tsne_comp}_{pca_comp}_{perplexity}_uncons_cons.png'
    color=3
    visualize_embeddings(uncons_cons_data, uncons_cons_labels, uncons_cons_label_names, color, graph_path,  rs, tsne_comp, pca_comp, perplexity)
