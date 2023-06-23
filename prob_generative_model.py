import numpy as np


def train_vae(vae, train_data, test_data, epochs, batch_size):
    x_train_size = train_data.shape[0]

    print(f"Train Dataset Size: {x_train_size} images")
    hist = vae.fit(x=train_data, epochs=epochs, batch_size=batch_size, verbose=1)

    return vae


def generate_vae(vae, target_data):
    x_train_size = target_data.shape[0]

    # print(f"Target Dataset Size: {x_train_size} images")
    _, _, z = vae.encoder.predict(target_data, verbose=0)
    recon_data = vae.decoder.predict(z)

    return recon_data


def generate_noisy_vae(vae, target_data, noise=None):
    x_train_size = target_data.shape[0]
    if noise is None:
        mean = 0
        std = 0.5
        noise = np.random.normal(mean, std, size=target_data.shape) * 1

    # print(f"Target Dataset Size: {x_train_size} images")
    _, _, z = vae.encoder.predict(target_data, verbose=0)
    noisy_z = (noise + z)
    recon_data = vae.decoder.predict(noisy_z, verbose=0)

    return recon_data