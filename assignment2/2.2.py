from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x,start_dim=-2))
])
data = datasets.MNIST('./data', transform=transform, download=True)
n, w, h = data.data.shape
data_flattened = data.data.numpy().reshape(len(data), -1)
data_normalized = data_flattened / 255.0

# PCA
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(data_normalized)

mean_vector = pca.mean_
explained_variance = pca.explained_variance_
principal_components = pca.components_.T

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=data.targets, alpha=0.5, cmap='tab10')
plt.colorbar(label='Digital label')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('PCA latent space')
plt.show()


def reconstruct_from_pca(pca,latent_space, n_samples=12, w=28, h=28):
    img = np.zeros((n_samples * w, n_samples * h))
    for i, y in enumerate(np.linspace(-2, 2, n_samples)):
        for j, x in enumerate(np.linspace(-2, 2, n_samples)):
            z = np.array([x, y]).reshape(1, -1)
            reconstructed = pca.inverse_transform(z)
            reconstructed_img = (reconstructed * 255.0).reshape(w, h)
            img[(n_samples - 1 - i) * w:(n_samples - 1 - i + 1) * w, j * h:(j + 1) * h] = reconstructed_img
    plt.imshow(img, cmap='gray')
    plt.title('PCA Reconstruction')
    plt.show()


reconstruct_from_pca(pca, latent_pca)

# Calculate reconstruction error
reconstruction_errors = []
components_range = range(2, 41)

for n_components in components_range:
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(data_normalized)  # Project data to lower dimensions
    reconstruction_error = np.mean((data_normalized - pca.inverse_transform(X_reduced)) ** 2)
    reconstruction_errors.append(reconstruction_error)

# Plot reconstruction error
plt.figure(figsize=(8, 5))
plt.plot(components_range, reconstruction_errors, marker='o', color='r', label='Reconstruction Error')
plt.xticks(ticks=components_range, labels=[str(x) for x in components_range])
plt.xlabel('Number of Eigenvectors')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Number of Principal Components')
plt.grid()
plt.legend()
plt.show()

# Calculate explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
xticks = range(1, len(explained_variance_ratio) + 1)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(xticks, explained_variance_ratio, marker='o', label='Cumulative Explained Variance')
plt.xticks(ticks=xticks, labels=[str(x) for x in xticks])
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Principal Components')
plt.grid()
plt.legend()
plt.show()


