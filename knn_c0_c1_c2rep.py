import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from classification_model import train_knn, predict_knn, predict_threshold_knn
from fetch_data import fetch_mnist_digit
from prob_generative_model import train_vae, generate_noisy_vae
from vae import vae_initializer
from sklearn.preprocessing import StandardScaler

# Model / data parameters
num_classes = 10
ndim = 28
input_shape = (ndim, ndim, 1)

(x_train, y_train), (x_test, y_test) = fetch_mnist_digit(ndim)


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

def random_samples(y_train, IMG_PER_CLASS):
    labeled_train_data_counter = np.zeros(10)
    labeled_train_data_indices = np.ones((10, IMG_PER_CLASS)) * -1
    rand_idx = np.random.randint(0, 10000, size=1000)

    for i, idx in enumerate(rand_idx):
        y_arg = y_train[idx]

        if labeled_train_data_counter.sum() == IMG_PER_CLASS * 10:
            break

        counter = int(labeled_train_data_counter[y_arg])
        if counter >= IMG_PER_CLASS:
            continue
        if labeled_train_data_indices[y_arg, counter] == -1:
            labeled_train_data_counter[y_arg] += 1
            labeled_train_data_indices[y_arg, counter] = idx

    labeled_train_data_indices = labeled_train_data_indices.astype(int)
    return labeled_train_data_indices


all_train_f1_scores, all_test_f1_scores = [], []
selftraining_success_f1_scores_per_iteration = []
selftraining_success_f1_scores_all = []

selected_labeled_indexes = random_samples(y_train, 10)
selected_labeled_indexes_flattened = selected_labeled_indexes.flatten()
x_initial = x_train[selected_labeled_indexes_flattened].copy()
y_initial = y_train[selected_labeled_indexes_flattened].copy()
clf = train_knn(x_initial, y_initial)
pred_y = predict_knn(clf, x_train)
c0_score = f1_score(pred_y, y_train, average='macro')
all_train_f1_scores.append(c0_score)
pred_y = predict_knn(clf, x_test)
c0_score = f1_score(pred_y, y_test, average='macro')
all_test_f1_scores.append(c0_score)

latent_dim = 128
epochs = 50
batch_size = 128
vae = vae_initializer(latent_dim=latent_dim)
vae_save_file = f"vae_mnist_d{latent_dim}_epochs{epochs}_batchsize{batch_size}_weights"
if os.path.exists(vae_save_file + ".index"):
    vae.load_weights(vae_save_file)
else:
    vae = train_vae(vae, train_data=x_train, test_data=x_test, epochs=epochs, batch_size=batch_size)
    vae.save_weights(vae_save_file)

# VAE
x_vae_labeled = []
y_vae_labeled = []
# Self-training
x_pseudo_labeled = []
pred_y_pseudo_labeled = []
true_y_pseudo_labeled = []

c1_surv_ratio = []
c2_surv_ratio = []
# Unlabeled
x_unlabeled = x_train[50000:].copy()
y_unlabeled = y_train[50000:].copy()
mean = 0
std = 0.5
for i in range(100):
    print(f"{y_unlabeled.shape[0]} unlabeled data are remained")
    if y_unlabeled.shape[0] == 0:
        break
    temp_x_vae_data = x_initial.copy()
    if len(x_vae_labeled) > 0:
        temp_x_vae_data = np.concatenate([temp_x_vae_data, x_vae_labeled], axis=0)
    if len(x_pseudo_labeled) > 0:
        temp_x_vae_data = np.concatenate([temp_x_vae_data, x_pseudo_labeled], axis=0)

    temp_y_vae_data = y_initial.copy()
    if len(y_vae_labeled) > 0:
        temp_y_vae_data = np.concatenate([temp_y_vae_data, y_vae_labeled])
    if len(pred_y_pseudo_labeled) > 0:
        temp_y_vae_data = np.concatenate([temp_y_vae_data, pred_y_pseudo_labeled])

    noise = np.random.normal(mean, std, size=(len(temp_x_vae_data), latent_dim)) * 1
    # Component 1
    recon_data = generate_noisy_vae(vae, temp_x_vae_data, noise)

    temp_x_clf_data = x_initial.copy()
    if len(x_vae_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_vae_labeled], axis=0)
    if len(x_pseudo_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_pseudo_labeled], axis=0)

    temp_y_clf_data = y_initial.copy()
    if len(y_vae_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, y_vae_labeled])
    if len(pred_y_pseudo_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, pred_y_pseudo_labeled])

    pred_y_pseudo_clf, survived_index = predict_threshold_knn(X_data=temp_x_clf_data, y_data=temp_y_clf_data,
                                                              X_target=recon_data, threshold=9)
    # if survived_index.shape[0] == 0:
    #     print("Nothing is survived and program is terminated")
    #     break
    # else:
    c1_surv_ratio.append(100*survived_index.shape[0]/recon_data.shape[0])
    print("Number of survived data after reconstructed images: ", survived_index.shape[0])
    if len(x_vae_labeled) == 0:
        x_vae_labeled = recon_data[survived_index, :].copy()
        y_vae_labeled = temp_y_vae_data[survived_index].copy()
    else:
        x_vae_labeled = np.concatenate([x_vae_labeled, recon_data[survived_index]], axis=0)
        y_vae_labeled = np.concatenate([y_vae_labeled, temp_y_vae_data[survived_index]])

    # Component 2 data prepration
    temp_x_clf_data = x_initial.copy()
    if len(x_vae_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_vae_labeled], axis=0)
    if len(x_pseudo_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_pseudo_labeled], axis=0)

    temp_y_clf_data = y_initial.copy()
    if len(y_vae_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, y_vae_labeled])
    if len(pred_y_pseudo_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, pred_y_pseudo_labeled])

    clf = train_knn(temp_x_clf_data, temp_y_clf_data)
    pred_y = predict_knn(clf, x_train)
    c1_train_score = f1_score(pred_y, y_train, average='macro')
    all_train_f1_scores.append(c1_train_score)
    pred_y = predict_knn(clf, x_test)
    c1_test_score = f1_score(pred_y, y_test, average='macro')
    all_test_f1_scores.append(c1_test_score)
    for j in range(10):
        # Component 2 run
        pred_y_pseudo_clf, survived_index = predict_threshold_knn(X_data=temp_x_clf_data, y_data=temp_y_clf_data,
                                                                  X_target=x_unlabeled[:100], threshold=9)
        if survived_index.shape[0] == 0:
            print("Nothing is survived and program is terminated")
            break
        else:
            c2_surv_ratio.append(100*survived_index.shape[0]/100)
            print("Number of survived data after unlabeled images: ", survived_index.shape[0])
        batch_success_score = f1_score(y_unlabeled[survived_index], pred_y_pseudo_clf, average='macro')
        selftraining_success_f1_scores_per_iteration.append(batch_success_score)

        if len(x_pseudo_labeled) == 0:
            x_pseudo_labeled = x_unlabeled[survived_index, :].copy()
            pred_y_pseudo_labeled = pred_y_pseudo_clf.copy()
            true_y_pseudo_labeled = y_unlabeled[survived_index].copy()
        else:
            x_pseudo_labeled = np.concatenate([x_pseudo_labeled, x_unlabeled[survived_index]], axis=0)
            pred_y_pseudo_labeled = np.concatenate([pred_y_pseudo_labeled, pred_y_pseudo_clf])
            true_y_pseudo_labeled = np.concatenate([true_y_pseudo_labeled, y_unlabeled[survived_index]])

        all_success_score = f1_score(true_y_pseudo_labeled, pred_y_pseudo_labeled, average='macro')
        selftraining_success_f1_scores_all.append(all_success_score)

        x_unlabeled = np.concatenate([x_unlabeled[100:], x_unlabeled[:100]], axis=0)
        y_unlabeled = np.concatenate([y_unlabeled[100:], y_unlabeled[:100]])

        x_unlabeled = np.delete(x_unlabeled, survived_index, axis=0)
        y_unlabeled = np.delete(y_unlabeled, survived_index)

    temp_x_clf_data = x_initial.copy()
    if len(x_vae_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_vae_labeled], axis=0)
    if len(x_pseudo_labeled) > 0:
        temp_x_clf_data = np.concatenate([temp_x_clf_data, x_pseudo_labeled], axis=0)

    temp_y_clf_data = y_initial.copy()
    if len(x_vae_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, y_vae_labeled])
    if len(pred_y_pseudo_labeled) > 0:
        temp_y_clf_data = np.concatenate([temp_y_clf_data, pred_y_pseudo_labeled])
    clf = train_knn(temp_x_clf_data, temp_y_clf_data)
    pred_y = predict_knn(clf, x_train)
    c2_train_score = f1_score(pred_y, y_train, average='macro')
    all_train_f1_scores.append(c2_train_score)
    print("Train f1-scores", all_train_f1_scores)
    plt.plot(all_train_f1_scores, label="Train")

    pred_y = predict_knn(clf, x_test)
    c2_test_score = f1_score(pred_y, y_test, average='macro')
    all_test_f1_scores.append(c2_test_score)
    plt.plot(all_test_f1_scores, label="Test")
    print("Test f1-scores", all_test_f1_scores)
    plt.xlabel("Iteration")
    plt.ylabel("F1-score")
    plt.title("Train and test data f1-score evaluation during SSL")
    plt.legend()
    plt.savefig("train_test_c0_c1_c2.png")
    plt.show()

    print("Success ratio for each batch of pseudo-data", selftraining_success_f1_scores_per_iteration)
    print("Success ratio for all pseudo-data", selftraining_success_f1_scores_all)
    plt.plot(selftraining_success_f1_scores_per_iteration, label="Each batch of pseudo-data")
    plt.plot(selftraining_success_f1_scores_all, label="All pseudo-data")
    plt.ylabel("Ratio (%)")
    plt.title("Success ratio")
    plt.legend()
    plt.show()


    plt.plot(c1_surv_ratio, label="Component 1")
    # plt.xlabel("Iteration")
    # plt.ylabel("Ratio (%)")
    # plt.title("Component 1 ratio per iteration")
    # plt.legend()
    # plt.show()
    plt.plot(c2_surv_ratio, label="Component 2")
    plt.xlabel("Iteration")
    plt.ylabel("Ratio (%)")
    plt.title("Component(s) ratio per iteration")
    plt.legend()
    plt.show()

np.savez("c0_c1_c2_mlp_results.npz", all_train_f1_scores=all_train_f1_scores, all_test_f1_scores=all_test_f1_scores,
         selftraining_success_f1_scores_per_iteration=selftraining_success_f1_scores_per_iteration,
         selftraining_success_f1_scores_all=selftraining_success_f1_scores_all)
