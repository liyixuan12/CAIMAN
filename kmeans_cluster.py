import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import random

import wandb 
import os
from tqdm import tqdm
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.cluster import KMeans
import joblib

from scipy.spatial import distance

from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# Shape of normalized training set cape_ml: (**, 501, 1501)
# Shape of normalized test set cape_ml: (**, 501, 1501)
# Reparameterization trick by sampling from a normal distribution
# Ensure reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_process_data(file_path):
    data = np.load(file_path)

    return data

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


        
# ELBO loss calculation
def elbo_loss(true, pred, z_mean, z_log_var):
    true = true.view(-1, 501 * 1501 * 6)

    # flatten
    x_mu = pred[:, :501 * 1501 * 6]
    x_log_var = pred[:, 501 * 1501 * 6:]

    x_mu = x_mu.type(torch.float64)
    x_log_var = x_log_var.type(torch.float64)

    # Gaussian reconstruction loss
    mse = -0.5 * torch.sum(torch.square(true - x_mu) / torch.exp(x_log_var), dim=1)
    var_trace = -0.5 * torch.sum(x_log_var, dim=1)
    log2pi = -0.5 * 501 * 1501 * 6 * np.log(2 * np.pi)
    log_likelihood = mse + var_trace + log2pi
    reconstruction_loss = -log_likelihood

    # KL divergence loss
    kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
    kl_loss = torch.sum(kl_loss, dim=1)
    kl_loss *= -0.5
    kl_loss = kl_loss.type(torch.float64)

    return -0.5 * (reconstruction_loss + kl_loss)


def kl_loss(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - z_mean.pow(2) - z_log_var.exp()
    kl_loss = torch.sum(kl_loss, dim=-1)
    kl_loss *= -0.5
    return kl_loss.mean()


def kl_reconstruction_loss(kl_weight):
    def _kl_reconstruction_loss(ouput, true):
        # flatten
        flattened_size = 501 * 1501 * 6
        z_mean, z_log_var, pred = ouput
        # reshape true and pred
        true = true.view(-1, flattened_size)
        x_mu = pred[:, :flattened_size]
        x_log_var = pred[:, flattened_size:]
        # print("Shape of true:", true.shape)
        # print("Shape of x_mu:", x_mu.shape)
        # print("Shape of x_log_var:", x_log_var.shape)

        # Gaussian reconstruction loss
        mse = -0.5 * torch.sum(torch.square(true - x_mu) / torch.exp(x_log_var), dim=1)
        var_trace = -0.5 * torch.sum(x_log_var, dim=1)
        log2pi = -0.5 * flattened_size * np.log(2 * np.pi)
        log_likelihood = mse + var_trace + log2pi
        reconstruction_loss = -log_likelihood

        # KL divergence loss
        kl_div_loss = kl_loss(z_mean, z_log_var)

        return torch.mean(reconstruction_loss + kl_weight * kl_div_loss)

    return _kl_reconstruction_loss


def categorical_crossentropy(y_true, y_pred):
    y_pred = F.log_softmax(y_pred, dim=1)  # Applying log_softmax
    return F.nll_loss(y_pred, y_true)  # Negative Log Likelihood Loss


def reconstruction(true, pred):
    # Shape for each time step: 501 * 1501 * 6
    flattened_size = 501 * 1501 * 6
    true = true.view(-1, flattened_size)

    x_mu = pred[:, :flattened_size]
    x_log_var = pred[:, flattened_size:]

    # Gaussian reshape
    mse = -0.5 * torch.sum(torch.square(true - x_mu) / torch.exp(x_log_var), dim=1)
    var_trace = -0.5 * torch.sum(x_log_var, dim=1)
    log2pi = -0.5 * flattened_size * np.log(2 * np.pi)
    log_likelihood = mse + var_trace + log2pi

    # print("log likelihood shape", log_likelihood.shape)

    return -log_likelihood.mean()


# Sampling layer
class Sampling(nn.Module):
    """Reparameterization trick by sampling from a normal distribution"""

    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_mean = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.bn_mean = nn.BatchNorm2d(64)
        self.conv_log_var = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.bn_log_var = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        
        # Linear layers for further dimensionality reduction
        self.fc1 = nn.Linear(12288, 6144)
        self.bn_fc1 = nn.BatchNorm1d(6144)
        self.fc2 = nn.Linear(6144, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.bn_fc3 = nn.BatchNorm1d(2048)
        # self.fc4 = nn.Linear(2048, 1024)
        # self.bn_fc4 = nn.BatchNorm1d(1024)

        # Final layers for mean and log variance
        # self.fc_mean = nn.Linear(1024, 1024)
        # self.fc_log_var = nn.Linear(1024, 1024)
        self.fc_mean = nn.Linear(2048, 2048)  # 改为2048
        self.fc_log_var = nn.Linear(2048, 2048)  # 改为2048

        self.sampling = Sampling()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print("After conv1:", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print("After conv2:", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("After conv3:", x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        #print("After conv4:", x.shape)
        x = F.relu(self.bn5(self.conv5(x)))
        z_mean = F.relu(self.bn_mean(self.conv_mean(x)))
        z_log_var = F.relu(self.bn_log_var(self.conv_log_var(x)))
        z_mean = self.flatten(z_mean)
        z_log_var = self.flatten(z_log_var)
       
        z_mean = F.relu(self.bn_fc1(self.fc1(z_mean)))
        z_log_var = F.relu(self.bn_fc1(self.fc1(z_log_var)))

        z_mean = F.relu(self.bn_fc2(self.fc2(z_mean)))
        z_log_var = F.relu(self.bn_fc2(self.fc2(z_log_var)))

        z_mean = F.relu(self.bn_fc3(self.fc3(z_mean)))
        z_log_var = F.relu(self.bn_fc3(self.fc3(z_log_var)))

        # z_mean = F.relu(self.bn_fc4(self.fc4(z_mean)))
        # z_log_var = F.relu(self.bn_fc4(self.fc4(z_log_var)))

        z_mean = self.fc_mean(z_mean)
        z_log_var = self.fc_log_var(z_log_var)
        print("After z_mean and flatten:", z_mean.shape)
        z = self.sampling(z_mean, z_log_var)
        print("After reparameterize:", z.shape)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2048, 512 * 16 * 47)
     
        self.conv_transpose1 = nn.ConvTranspose2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.conv_transpose2a = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.conv_transpose3a = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_transpose4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)

        self.conv_transpose4a = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, output_padding=0)
        self.conv_transpose_x_mu = nn.ConvTranspose2d(128, 6, kernel_size=(3, 3), stride=(1, 1), padding=1, output_padding=0)
        self.conv_transpose_log_var = nn.ConvTranspose2d(128, 6, kernel_size=(3, 3), stride=(1, 1), padding=1, output_padding=0)
        
        self.adaptive_pool = nn.AdaptiveMaxPool2d((501, 1501))

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 16, 47)
        print(f"Shape after reshape: {x.shape}")
        x = F.relu(self.conv_transpose1(x))
        print(f"Shape after conv_transpose1: {x.shape}")
        x = F.relu(self.conv_transpose2(x))
        print(f"Shape after conv_transpose2: {x.shape}")
        x = F.relu(self.conv_transpose2a(x))
        print(f"Shape after conv_transpose2a: {x.shape}")
        x = F.relu(self.conv_transpose3(x))
        print(f"Shape after conv_transpose3: {x.shape}")
        x = F.relu(self.conv_transpose3a(x))
        print(f"Shape after conv_transpose3a: {x.shape}")
        x = F.relu(self.conv_transpose4(x))
        print(f"Shape after conv_transpose4: {x.shape}")
        x = F.relu(self.conv_transpose4a(x))
        print(f"Shape after conv_transpose4a: {x.shape}")
        x_mu = F.sigmoid(self.conv_transpose_x_mu(x))
        print(f"Shape after conv_transpose_x_mu: {x_mu.shape}")
        x_log_var = self.conv_transpose_log_var(x)
        print(f"Shape after conv_transpose_log_var: {x_log_var.shape}")
        x_mu = self.adaptive_pool(x_mu)
        print(f"Shape after adaptive_pool_x_mu: {x_mu.shape}")
        x_log_var = self.adaptive_pool(x_log_var)
        print(f"Shape after adaptive_pool_x_log_var: {x_log_var.shape}")
        #x_mu = x_mu[:, :, 1:-1, :-1]  # Cropping equivalent
        # x_log_var = x_log_var[:, :, 1:-1, :-1]  # Cropping equivalent
        x_mu = x_mu.reshape(x_mu.shape[0], -1)
        x_log_var = x_log_var.reshape(x_log_var.shape[0], -1)
        x_mu_log_var = torch.cat([x_mu, x_log_var], dim=1)
        print(f"Final shape before return: {x_mu_log_var.shape}")
        return x_mu_log_var


# Variational Autoencoder (VAE) model
class VAEModel(nn.Module):
    def __init__(self, input_channels):
        super(VAEModel, self).__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder()

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        pred = self.decoder(z)
        return z_mean, z_log_var, z, pred


# Train VAE model
def train_vae(vae_model, train_loader, test_loader, optimizer, criterion, epochs, kl_weight, scheduler, early_stopping):
  
    train_latent_vectors = []
    test_latent_vectors = []
    vae_model.train()
    train_loss_list = []
    train_kl_loss_list = []
    train_reconstruction_loss_list = []

    test_loss_list = []
    test_kl_loss_list = []
    test_reconstruction_loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        total_kl = 0
        total_reconstruction_loss = 0
        loop = tqdm(train_loader, total=len(train_loader))
        for inputs, _ in loop:
            # Forward pass
            inputs=inputs.to(device)
            optimizer.zero_grad()
            z_mean, z_log_var, z, pred = vae_model(inputs)

            # Calculate loss
            
            kl = kl_loss(z_mean, z_log_var)
            reconstruction_loss = reconstruction(inputs, pred)
            
            loss = criterion((z_mean, z_log_var, pred), inputs)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_kl += kl.item()
            total_reconstruction_loss += reconstruction_loss.item()
            loop.set_description(f'vae[{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=loss.item())

            train_latent_vectors.append(z.detach().cpu())

        train_loss_list.append((total_loss / len(train_loader)))
        train_kl_loss_list.append(total_kl / len(train_loader))
        train_reconstruction_loss_list.append(total_reconstruction_loss / len(train_loader))
        # Test
        scheduler.step(total_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], train Loss: {total_loss / len(train_loader):.4f}')

        vae_model.eval()

        total_loss = 0.
        total_kl = 0
        total_reconstruction_loss = 0
        for inputs, _ in test_loader:
            # Forward pass
            inputs=inputs.to(device)

            z_mean, z_log_var, z, pred = vae_model(inputs)

            # Calculate loss
            loss = criterion((z_mean, z_log_var, pred), inputs)
            kl = kl_loss(z_mean, z_log_var)
            total_kl += kl.item()
            reconstruction_loss = reconstruction(inputs, pred)
            total_reconstruction_loss += reconstruction_loss.item()
            # Backward pass and optimize
            total_loss += loss.item()

            test_latent_vectors.append(z.detach().cpu())

        test_loss_list.append((total_loss / len(test_loader)))
        test_kl_loss_list.append(total_kl / len(test_loader))
        test_reconstruction_loss_list.append(total_reconstruction_loss / len(test_loader))
        print(f'Epoch [{epoch + 1}/{epochs}], test Loss: {total_loss / len(test_loader):.4f}')
        train_latent_vectors = torch.cat(train_latent_vectors, dim=0)
        test_latent_vectors = torch.cat(test_latent_vectors, dim=0)
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_kl_loss": total_kl / len(train_loader),
            "train_reconstruction_loss": total_reconstruction_loss / len(train_loader),
            "test_loss": total_loss / len(test_loader),
            "test_kl_loss": total_kl / len(test_loader),
            "test_reconstruction_loss": total_reconstruction_loss / len(test_loader)
        })
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return (train_loss_list, train_kl_loss_list, train_reconstruction_loss_list), (
    test_loss_list, test_kl_loss_list, test_reconstruction_loss_list), train_latent_vectors, test_latent_vectors

def calculate_data_counts(base_dir, training_months):
    data_counts = []
    for month in training_months:
        file_path = os.path.join(base_dir, f"{month}_2011_omega.npy")
        try:
            data = np.load(file_path)
            data_counts.append(data.shape[0])  # 假设数据点存储在第一个维度
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            data_counts.append(0)  # 如果文件不存在，计数为0
    return data_counts


def perform_kmeans(z, n_clusters, original_indices):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(z)
    
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < n_clusters:
        print(f"Warning: Number of unique clusters ({len(unique_labels)}) is less than expected ({n_clusters}).")

    print(f"Unique clusters: {unique_labels}")
    
    joblib.dump(kmeans, f'km_{n_clusters}.model')
    
    with open(f'result_{n_clusters}.txt', 'w', encoding='utf-8') as f:
        f.write("Cluster centers:\n")
        for index, center in enumerate(kmeans.cluster_centers_):
            f.write(f"Center {index + 1}: {center}\n")
        f.write("\nLabels:\n")
        f.write(' '.join(map(str, labels)))

    cluster_closest_points = {}
    for i in range(n_clusters):
        distances = distance.cdist(z, [kmeans.cluster_centers_[i]], 'euclidean').flatten()
        nearest_indices = np.argsort(distances)[:10]
        nearest_indices = np.unique(nearest_indices)  
        nearest_original_indices = original_indices[nearest_indices]
        cluster_closest_points[i] = nearest_original_indices
        
        print(f"Cluster {i + 1}, nearest 10 original indices:")
        for idx in nearest_original_indices:
            print(idx)

    return kmeans, cluster_closest_points

def analyze_clusters(closest_points, original_data, latent_vectors, output_file, original_indices):
    cluster_details = {}
    with open(output_file, 'w') as f:
        for cluster_id, indices in closest_points.items():
            f.write(f"Cluster {cluster_id}: Nearest Original Indices: {indices}\n")
            cluster_details[cluster_id] = []
            for idx in indices:
                if idx >= len(original_indices):
                    f.write(f"Index {idx} is out of bounds for original_data with size {len(original_data)}.\n")
                    continue
                original_idx = original_indices[idx]
                details = {"Index": idx, "Coordinates": (latent_vectors[idx, 0], latent_vectors[idx, 1])}
                f.write(f"Sample {idx}, Coordinates: {details['Coordinates']}\n")
                for channel in range(original_data.shape[1]):
                    center_height = original_data.shape[2] // 2
                    center_width = original_data.shape[3] // 2
                    pixel_value = original_data[idx, channel, center_height, center_width]
                    details[f"Channel_{channel}_Center_Pixel_Value"] = pixel_value
                    f.write(f"Sample {idx}, Channel {channel}, Center Pixel Value: {pixel_value}\n")
                cluster_details[cluster_id].append(details)
            f.write("\n")
    return cluster_details


def plot_clusters(latent_vectors, n_clusters):
    # use kmeans to cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(latent_vectors)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    closest_data = []
    closest_indices = []
    for center in centers:
        distances = distance.cdist(latent_vectors, [center], 'euclidean').flatten()
        nearest_indices = np.argsort(distances)[:10]
        closest_data.append(latent_vectors[nearest_indices])
        closest_indices.append(nearest_indices)
    
    return labels, centers, closest_data, closest_indices, kmeans


def visualize_clusters(latent_vectors, original_data, n_clusters):
    labels, centers, closest_data, closest_indices, kmeans = plot_clusters(latent_vectors, n_clusters)
    num_channels = original_data.shape[1]  

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, alpha=0.6, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', label='Centers')

    for i, data in enumerate(closest_data):
        plt.scatter(data[:, 0], data[:, 1], s=100, edgecolors='black', label=f'Closest to Center {i+1}' if i == 0 else "")

    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Cluster Visualization with Centers and Closest Points')
    plt.legend()
    plt.grid(True)
    plt.savefig("Laten_space_z_clusters.png")
    plt.show()
    print(f"Cluster centers: {centers}")

def perform_gmm(z, n_clusters, original_indices):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10).fit(z)
    labels = gmm.predict(z)
    unique_labels = np.unique(labels)
    if len(unique_labels) < n_clusters:
        print(f"Warning: Number of unique clusters ({len(unique_labels)}) is less than expected ({n_clusters}).")
    print(f"Unique clusters: {unique_labels}")
    joblib.dump(gmm, f'gmm_{n_clusters}.model')
    with open(f'result_{n_clusters}.txt', 'w', encoding='utf-8') as f:
        f.write("Cluster centers:\n")
        for index, center in enumerate(gmm.means_):
            f.write(f"Center {index + 1}: {center}\n")
        f.write("\nLabels:\n")
        f.write(' '.join(map(str, labels)))
    cluster_closest_points = {}
    for i in range(n_clusters):
        distances = distance.cdist(z, [gmm.means_[i]], 'euclidean').flatten()
        nearest_indices = np.argsort(distances)[:10]
        nearest_indices = np.unique(nearest_indices)
        nearest_original_indices = original_indices[nearest_indices]
        cluster_closest_points[i] = nearest_original_indices
        print(f"Cluster {i + 1}, nearest 10 original indices:")
        for idx in nearest_original_indices:
            print(idx)
    return gmm, cluster_closest_points

def check_nearest_points_same_cluster(kmeans, z, cluster_center_idx, num_points):
    distances = distance.cdist(z, [kmeans.cluster_centers_[cluster_center_idx]], 'euclidean').flatten()
    nearest_indices = np.argsort(distances)[:num_points]
    nearest_indices = np.unique(nearest_indices)  
    nearest_labels = kmeans.labels_[nearest_indices]
    center_label = np.bincount(nearest_labels).argmax()
    all_same_cluster = np.all(nearest_labels == center_label)
    not_in_same_cluster = nearest_indices[nearest_labels != center_label]
    return nearest_indices, nearest_labels, all_same_cluster, not_in_same_cluster

def plot_latent_space(latent_vectors, method, save_path):
    """
    Visualize the latent space using PCA or t-SNE.
    
    Parameters:
    latent_vectors (numpy.ndarray): The latent vectors to visualize.
    method (str): The method to use for visualization ('pca' or 'tsne').
    save_path (str): The file path to save the plot image. If None, the image will not be saved.
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    reduced_latent_vectors = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_latent_vectors[:, 0], y=reduced_latent_vectors[:, 1], alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()
def evaluate_clustering_performance(kmeans_model, latent_vectors):
    labels = kmeans_model.labels_
    score = silhouette_score(latent_vectors, labels)
    return score

def main():
    #set_seed(42)
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="caiman", entity="chi_li")
    base_dir = '/data'
    training_months = ['Jan2011', 'Feb2011', 'March2011', 'April2011', 'May2011', 'June2011', 'July2011', 'August2011', 'September2011']
    testing_months = ['October2011', 'November2011', 'December2011']
    training_files = [f"{month}_2011_omega.npy" for month in training_months]
    testing_files = [f"{month}_2011_omega.npy" for month in testing_months]
    train_data = []
    for train_file in training_files:
        month_data = load_and_process_data(os.path.join(base_dir, train_file))
        train_data.append(month_data)
    if train_data:
        train_data = np.concatenate(train_data, axis=0)
    test_data = []
    for test_file in testing_files:
        month_data = load_and_process_data(os.path.join(base_dir, test_file))
        test_data.append(month_data)
    if test_data:
        test_data = np.concatenate(test_data, axis=0)
    original_data = np.copy(train_data)
    data_counts = calculate_data_counts(base_dir, training_months)
    for month, count in zip(training_months, data_counts):
        print(f"{month} has {count} data points.")
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    vae_train_dataset = TensorDataset(train_data, train_data)
    vae_train_loader = DataLoader(vae_train_dataset, batch_size=8, shuffle=True)
    vae_test_dataset = TensorDataset(test_data, test_data)
    vae_test_loader = DataLoader(vae_test_dataset, batch_size=8, shuffle=False)
    epochs = 5
    initial_kl_weight = 0.01
    max_kl_weight = 1.0
    vae_model = VAEModel(input_channels=6)
    vae_model = nn.DataParallel(vae_model)
    vae_model.to(device)
    optimizer_vae = optim.Adam(vae_model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer_vae, mode='min', factor=0.1, patience=5, verbose=True)
    n_clusters = 2
    all_train_latent_vectors = []
    all_indices = []
    for epoch in range(epochs):
        kl_weight = initial_kl_weight + (max_kl_weight - initial_kl_weight) * epoch / epochs
        criterion_vae = kl_reconstruction_loss(kl_weight)
        _, _, latent_vectors, _ = train_vae(vae_model, vae_train_loader, vae_test_loader, optimizer_vae, criterion_vae, 1, kl_weight, scheduler, early_stopping)
        latent_vectors = latent_vectors.detach().cpu().numpy()
        if len(latent_vectors.shape) > 2:
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
        all_train_latent_vectors.extend(latent_vectors)
        all_indices.extend(np.arange(len(latent_vectors)))
    all_train_latent_vectors = np.vstack(all_train_latent_vectors)
    all_indices = np.array(all_indices)
    save_dir = 'project/project_caiman/save/saveImg'
    os.makedirs(save_dir, exist_ok=True)
    plot_latent_space(all_train_latent_vectors, method='pca', save_path=os.path.join(save_dir, 'latent_space_pca.png'))
    plot_latent_space(all_train_latent_vectors, method='tsne', save_path=os.path.join(save_dir, 'latent_space_tsne.png'))
    kmeans_model, closest_points = perform_kmeans(all_train_latent_vectors, n_clusters, all_indices)
    output_file = "cluster_analysis.txt"
    cluster_details = analyze_clusters(closest_points, original_data, all_train_latent_vectors, output_file, all_indices)
    print("Cluster details saved to:", output_file)
    visualize_clusters(all_train_latent_vectors, original_data, n_clusters)
    selected_clusters = [0, 1]
    num_points = 10
    for cluster_center_idx in selected_clusters:
        nearest_indices, nearest_labels, all_same_cluster, not_in_same_cluster = check_nearest_points_same_cluster(kmeans_model, all_train_latent_vectors, cluster_center_idx, num_points)
        print(f"Cluster center {cluster_center_idx}:")
        print(f"All nearest points belong to the same cluster: {all_same_cluster}")
        if not all_same_cluster:
            print(f"Points not belonging to cluster {cluster_center_idx}: {not_in_same_cluster}")
        else:
            print(f"Nearest 10 points indices: {nearest_indices}")
            print(f"Labels of nearest points: {nearest_labels}")
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}:', torch.cuda.get_device_properties(i))
        print('Allocated:', torch.cuda.memory_allocated(i) / 1024**3, 'GB')
        print('Cached:   ', torch.cuda.memory_reserved(i) / 1024**3, 'GB')



def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int,
                        help='This option specifies the config file to use to construct and train the VAE.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(device)
    main()