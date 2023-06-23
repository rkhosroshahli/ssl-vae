import numpy as np
import matplotlib.pyplot as plt

def plot_multi(hist_clf_train_f1, hist_clf_test_f1, title, xlabel="Iteration", ylabel="F1-score", save_link=""):
    plt.plot(hist_clf_train_f1, label="Train")
    plt.plot(hist_clf_test_f1, label="Test")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(save_link)

def plot_single(data, title, xlabel="Iteration", ylabel="F1-score", save_link=""):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(save_link)

# npz_link = "c0_c1_c2_mlp_results.npz"
npz_link = "c0_c2_mlp_results.npz"
npz_file = np.load(npz_link)
all_train_f1_scores=npz_file['all_train_f1_scores']
all_test_f1_scores=npz_file['all_test_f1_scores']

plot_multi(all_train_f1_scores, all_test_f1_scores, "Train and test F1-score per iteration", xlabel="Iteration", ylabel="F1-score", save_link="train_test_plot")


selftraining_success_f1_scores_per_iteration=npz_file['selftraining_success_f1_scores_per_iteration']
selftraining_success_f1_scores_all=npz_file['selftraining_success_f1_scores_all']

plot_single(selftraining_success_f1_scores_per_iteration, "Pseudo batch success ratio F1-score per iteration", xlabel="Iteration", ylabel="F1-score", save_link="success_ratio_each_pseudo_batch")
plot_single(selftraining_success_f1_scores_all, "Accumulative pseudo data success ratio F1-score per iteration", xlabel="Iteration", ylabel="F1-score", save_link="success_ratio_accumulative")