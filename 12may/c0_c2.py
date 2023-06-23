import numpy as np
from sklearn.metrics import f1_score

from classification_model import train_mlp, predict_mlp, predict_threshold_mlp
from fetch_data import fetch_mnist_digit
from prob_generative_model import train_vae, generate_noisy_vae
from vae import vae_initializer

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
clf = train_mlp(x_initial, y_initial)
pred_y = predict_mlp(clf, x_train)
c0_score = f1_score(pred_y, y_train, average='macro')
all_train_f1_scores.append(c0_score)
pred_y = predict_mlp(clf, x_test)
c0_score = f1_score(pred_y, y_test, average='macro')
all_test_f1_scores.append(c0_score)

# latent_dim = 4
# vae = init_vae(latent_dim=latent_dim)
# vae = train_vae(vae, train_data=x_train, test_data=x_test, epochs=50, batch_size=128)

# VAE
x_vae_labeled = []
y_vae_labeled = []
# Self-training
x_pseudo_labeled = []
pred_y_pseudo_labeled = []
true_y_pseudo_labeled = []
# Unlabeled
x_unlabeled = x_train[50000:].copy()
y_unlabeled = y_train[50000:].copy()
mean = 0
std = 0.5
for i in range(1000):
    print(f"Iteration. {i}, remaining unlabeled data size. {y_unlabeled.shape[0]}")
    if y_unlabeled.shape[0] == 0:
        print(x_unlabeled.shape[0])
        break
    # temp_x_vae_data = x_initial.copy()
    # if len(x_vae_labeled) > 0:
    #     temp_x_vae_data = np.concatenate([temp_x_vae_data, x_vae_labeled], axis=0)
    # if len(x_pseudo_labeled) > 0:
    #     temp_x_vae_data = np.concatenate([temp_x_vae_data, x_pseudo_labeled], axis=0)
    #
    # temp_y_vae_data = y_initial.copy()
    # if len(y_vae_labeled) > 0:
    #     temp_y_vae_data = np.concatenate([temp_y_vae_data, y_vae_labeled])
    # if len(pred_y_pseudo_labeled) > 0:
    #     temp_y_vae_data = np.concatenate([temp_y_vae_data, pred_y_pseudo_labeled])
    #
    # noise = np.random.normal(mean, std, size=(len(temp_x_vae_data), latent_dim)) * 1
    # # Component 1
    # recon_data = generate_noisy_vae(vae, temp_x_vae_data, noise)
    # if len(x_vae_labeled) == 0:
    #     x_vae_labeled = recon_data.copy()
    #     y_vae_labeled = temp_y_vae_data.copy()
    # else:
    #     x_vae_labeled = np.concatenate([x_vae_labeled, recon_data], axis=0)
    #     y_vae_labeled = np.concatenate([y_vae_labeled, temp_y_vae_data])
    #
    # temp_x_clf_data = x_initial.copy()
    # if len(x_vae_labeled) > 0:
    #     temp_x_clf_data = np.concatenate([temp_x_clf_data, x_vae_labeled], axis=0)
    # if len(x_pseudo_labeled) > 0:
    #     temp_x_clf_data = np.concatenate([temp_x_clf_data, x_pseudo_labeled], axis=0)
    #
    # temp_y_clf_data = y_initial.copy()
    # if len(x_vae_labeled) > 0:
    #     temp_y_clf_data = np.concatenate([temp_y_clf_data, y_vae_labeled])
    # if len(pred_y_pseudo_labeled) > 0:
    #     temp_y_clf_data = np.concatenate([temp_y_clf_data, pred_y_pseudo_labeled])
    #
    # clf = train_mlp(temp_x_clf_data, temp_y_clf_data)
    # pred_y = predict_mlp(clf, x_train)
    # c1_score = f1_score(pred_y, y_train, average='macro')
    # all_train_f1_scores.append(c1_score)
    # pred_y = predict_mlp(clf, x_test)
    # c1_score = f1_score(pred_y, y_test, average='macro')
    # all_test_f1_scores.append(c1_score)
    # Component 2
    pred_y_pseudo_clf, survived_index = predict_threshold_mlp(clf, x_unlabeled[:100], threshold=0.99)
    if len(survived_index) == 0:
        break
    selftraining_success_f1_scores_per_iteration.append(
        f1_score(y_unlabeled[survived_index], pred_y_pseudo_clf, average='macro'))
    if len(x_pseudo_labeled) == 0:
        x_pseudo_labeled = x_unlabeled[survived_index].copy()
        pred_y_pseudo_labeled = pred_y_pseudo_clf
        true_y_pseudo_labeled = y_unlabeled[survived_index].copy()
    else:
        x_pseudo_labeled = np.concatenate([x_pseudo_labeled, x_unlabeled[survived_index]], axis=0)
        pred_y_pseudo_labeled = np.concatenate([pred_y_pseudo_labeled, pred_y_pseudo_clf])
        true_y_pseudo_labeled = np.concatenate([true_y_pseudo_labeled, y_unlabeled[survived_index]])
    selftraining_success_f1_scores_all.append(f1_score(true_y_pseudo_labeled, pred_y_pseudo_labeled, average='macro'))
    x_unlabeled = np.concatenate([x_unlabeled, np.delete(x_unlabeled[:100], survived_index, axis=0)], axis=0)
    y_unlabeled = np.concatenate([y_unlabeled, np.delete(y_unlabeled[:100], survived_index)])
    x_unlabeled = x_unlabeled[100:]
    y_unlabeled = y_unlabeled[100:]

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
    clf = train_mlp(temp_x_clf_data, temp_y_clf_data)
    pred_y = predict_mlp(clf, x_train)
    c2_score = f1_score(pred_y, y_train, average='macro')
    all_train_f1_scores.append(c2_score)
    pred_y = predict_mlp(clf, x_test)
    c2_score = f1_score(pred_y, y_test, average='macro')
    all_test_f1_scores.append(c2_score)

np.savez("c0_c2_mlp_results.npz", all_train_f1_scores=all_train_f1_scores, all_test_f1_scores=all_test_f1_scores,
         selftraining_success_f1_scores_per_iteration=selftraining_success_f1_scores_per_iteration,
         selftraining_success_f1_scores_all=selftraining_success_f1_scores_all)
