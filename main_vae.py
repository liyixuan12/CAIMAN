import joblib
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.cluster import KMeans  
import os
import glob
import xarray as xr

from tqdm import tqdm
  
import wandb
import argparse

import matplotlib.pyplot as plt
import seaborn as sns



# Shape of normalized training set cape_ml: (96, 501, 1501)
# Shape of normalized test set cape_ml: (24, 501, 1501)
# Reparameterization trick by sampling from a normal distribution
def reparameterize(z_mean, z_log_var):
    batch, dim = z_mean.size()
    epsilon = torch.randn(batch, dim).to(z_mean.device)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# ELBO loss calculation
def elbo_loss(true, pred, z_mean, z_log_var):
    true = true.view(-1, 501 * 1501 * 3)

    # flatten
    x_mu = pred[:, :501 * 1501 * 3]
    x_log_var = pred[:, 501 * 1501 * 3:]

    x_mu = x_mu.type(torch.float64)
    x_log_var = x_log_var.type(torch.float64)

    # Gaussian reconstruction loss
    mse = -0.5 * torch.sum(torch.square(true - x_mu) / torch.exp(x_log_var), dim=1)
    var_trace = -0.5 * torch.sum(x_log_var, dim=1)
    log2pi = -0.5 * 501 * 1501 * 3 * np.log(2 * np.pi)
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
        flattened_size = 501 * 1501 * 3
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
    # Shape for each time step: 501 * 1501 * 2
    flattened_size = 501 * 1501 * 3
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
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv_mean = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.conv_log_var = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.sampling = Sampling()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = F.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = F.relu(self.conv3(x))
        print("After conv3:", x.shape)
        x = F.relu(self.conv4(x))
        print("After conv4:", x.shape)
        z_mean = self.flatten(F.relu(self.conv_mean(x)))
        print("After conv_mean and flatten:", z_mean.shape)
        z_log_var = self.flatten(F.relu(self.conv_log_var(x)))
        print("After conv_log_var and flatten:", z_log_var.shape)
        z = self.sampling(z_mean, z_log_var)
        print("After sampling:", z.shape)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.dense = nn.Linear(42 * 21 * 64, 42 * 84 * 1 * 512)
        self.reshape = lambda x: x.view(-1, 64, 32, 94)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                  output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                  output_padding=1)
        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                  output_padding=1)
        self.conv_transpose_x_mu = nn.ConvTranspose2d(64, 2, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                      output_padding=1)
        self.conv_transpose_log_var = nn.ConvTranspose2d(64, 2, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                         output_padding=1)
        #
        self.adaptive_pool = nn.AdaptiveAvgPool2d((501, 1501))

    def forward(self, x):
        x = self.reshape(x)
        print(f"Shape after reshape: {x.shape}")
        x = F.relu(self.conv_transpose1(x))
        print(f"Shape after conv_transpose1: {x.shape}")
        x = F.relu(self.conv_transpose2(x))
        print(f"Shape after conv_transpose2: {x.shape}")
        x = F.relu(self.conv_transpose3(x))
        print(f"Shape after conv_transpose3: {x.shape}")
        x = F.relu(self.conv_transpose4(x))
        print(f"Shape after conv_transpose4: {x.shape}")
        x_mu = F.relu(self.conv_transpose_x_mu(x))
        print(f"Shape after conv_transpose_x_mu: {x_mu.shape}")
        x_log_var = F.relu(self.conv_transpose_log_var(x))
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
        z_mean, z_log_var, x = self.encoder(x)
        x = self.decoder(x)
        return z_mean, z_log_var, x



# Plot training losses for VAE
def plot_training_losses(history):
    (train_total_loss, train_kl_loss, train_reconstruction), (
    valid_total_loss, valid_kl_loss, valid_reconstruction) = history
    epochs = range(1, len(train_reconstruction) + 1)

    plt.figure(figsize=(12.8, 4.8))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_total_loss, 'b', label='Train Total Loss')
    plt.plot(epochs, valid_total_loss, 'r', label='Valid Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_kl_loss, 'b', label='Train KL Loss')
    plt.plot(epochs, valid_kl_loss, 'r', label='Valid KL Loss')
    plt.title('KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_reconstruction, 'b', label='Train Reconstruction Loss')
    plt.plot(epochs, valid_reconstruction, 'r', label='Valid Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_losses_vae.png')


# Train VAE model
def train_vae(vae_model, train_loader, test_loader, optimizer, criterion, epochs):
    wandb.init()
    vae_model.train()
    train_loss_list = []
    train_kl_loss_list = []
    train_reconstruction_loss_list = []

    test_loss_list = []
    test_loss_list = []
    test_kl_loss_list = []
    test_reconstruction_loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        total_kl = 0
        total_reconstruction_loss = 0
        loop = tqdm(train_loader, total=len(train_loader))
        for inputs, labels in loop:
            # Forward pass
            outputs = vae_model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            kl = kl_loss(outputs[0], outputs[1])
            total_kl += kl.item()
            reconstruction_loss = reconstruction(labels, outputs[2])
            total_reconstruction_loss += reconstruction_loss.item()
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f'vae[{epoch + 1}/{epochs}]')
            loop.set_postfix(losss=loss.item())
        train_loss_list.append((total_loss / len(train_loader)))
        train_kl_loss_list.append(total_kl / len(train_loader))
        train_reconstruction_loss_list.append(total_reconstruction_loss / len(train_loader))
        # Test

        total_loss = 0.
        total_kl = 0
        total_reconstruction_loss = 0
        for inputs, labels in test_loader:
            # Forward pass
            outputs = vae_model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            kl = kl_loss(outputs[0], outputs[1])
            total_kl += kl.item()
            reconstruction_loss = reconstruction(labels, outputs[2])
            total_reconstruction_loss += reconstruction_loss.item()

            total_loss += loss.item()
        test_loss_list.append((total_loss / len(test_loader)))
        test_kl_loss_list.append(total_kl / len(test_loader))
        test_reconstruction_loss_list.append(total_reconstruction_loss / len(test_loader))
        print(f'Epoch [{epoch + 1}/{epochs}], train Loss: {total_loss / len(train_loader):.4f}')
        print(f'Epoch [{epoch + 1}/{epochs}], test Loss: {total_loss / len(test_loader):.4f}')
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_kl_loss": total_kl / len(train_loader),
            "train_reconstruction_loss": total_reconstruction_loss / len(train_loader),
            "test_loss": total_loss / len(test_loader),
            "test_kl_loss": total_kl / len(test_loader),
            "test_reconstruction_loss": total_reconstruction_loss / len(test_loader)
        })
    return (train_loss_list, train_kl_loss_list, train_reconstruction_loss_list), (
    test_loss_list, test_kl_loss_list, test_reconstruction_loss_list)


# Get middle output from the encoder
def get_middle_output(encoder, data_loader):
    """Get middle layer output from the encoder"""
    encoder.eval()  # Set to evaluation modehj
    outputs = []

    with torch.no_grad():  # Do not compute gradient
        for data in data_loader:
            if isinstance(data, tuple) or isinstance(data, list):
                # If data_loader returns a tuple of data and labels, only take the data part
                data = data[0]
            output = encoder(data)[2]  # Assume the third output is the middle layer output
            outputs.append(output)

    return torch.cat(outputs, dim=0)  # Concatenate all batch outputs



def plot_pic(array, index):
    plt.imshow(array)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{index}.png')
    plt.show()

def train_and_get_hidden_state(train_data, test_data, input_channels, init_model_path=None,save_model_path=None,save_file_name='feature1.pt'):
    #feature  = torch.randn((120, 6 , 501, 1501), dtype=torch.float32)
    #feature = torch.tensor(data, dtype=torch.float32)
    # train_feature = feature[:96,:,:,:]
    # test_feature = feature[96:,:,:,:]

    # 转换为 PyTorch 张量
    train_feature = torch.tensor(train_data, dtype=torch.float32)
    test_feature = torch.tensor(test_data, dtype=torch.float32)
    vae_train_dataset = TensorDataset(train_feature, train_feature)
    vae_train_loader = DataLoader(vae_train_dataset, batch_size=1, shuffle=True)
    vae_test_dataset = TensorDataset(test_feature, test_feature)
    vae_test_loader = DataLoader(vae_test_dataset, batch_size=1, shuffle=False)
    
    vae_model = VAEModel(input_channels=6)
    if init_model_path==None:
        optimizer_vae = optim.Adam(vae_model.parameters(), lr=1e-3)
        criterion_vae = kl_reconstruction_loss(0.5)
        train_vae(vae_model,vae_train_loader,vae_test_loader,optimizer_vae,criterion_vae,epochs=1)
        torch.save(vae_model.state_dict(), save_model_path)
    else:
        vae_model.load_state_dict(torch.load(init_model_path))
    
    vae_model.eval()
    hidden_feature_list = []
    with torch.no_grad():
        for data in torch.cat([train_feature, test_feature], dim=0):
            #_,_,hidden_feature = vae_model.encoder(feature[i:i+1])
            _, _, hidden_feature = vae_model.encoder(data.unsqueeze(0))
            hidden_feature_list.append(hidden_feature)

    hidden_feature = torch.cat(hidden_feature_list,dim=0)
    torch.save(hidden_feature, save_file_name)
    
def train_k_means(file1,file2,file3):
    hidden_feature1 = torch.load(file1)
    hidden_feature2 = torch.load(file2)
    hidden_feature3 = torch.load(file3)
    
    hidden_feature = torch.cat([hidden_feature1, hidden_feature2, hidden_feature3], dim=1)
    hidden_feature = torch.split(hidden_feature,1,dim=0)
    hidden_feature =[i.squeeze(0).numpy().tolist() for i in hidden_feature]
    # train_hidden_feature = hidden_feature[:96]
    # test_hidden_feature = hidden_feature[96:]
    
    # 2 cluster
    kmeans = KMeans(n_clusters=2, random_state=0).fit(hidden_feature)  
    
    # 输出聚类中心和标签  
    print("Cluster centers:")  
    print(kmeans.cluster_centers_)  
    print("Labels:")  
    print(kmeans.labels_)  
    joblib.dump(kmeans,'km.model')
    with open('result.txt','w',encoding='utf-8') as f:
        for i in kmeans.labels_:
            f.write(str(i)+'\n')

    pass 

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int,
                        help='This option specifies the config file to use to construct and train the VAE.')

    args = parser.parse_args()
    return args


def load_and_normalize_data(month_dir, min_val, max_val):
    """加载数据，并使用提供的最小最大值进行归一化。返回归一化数据列表。"""
    nc_files = glob.glob(os.path.join(month_dir, '*.nc'))
    normalized_data = []
    for file_path in tqdm(nc_files, desc=f"Processing files in {month_dir}"):
        with xr.open_dataset(file_path) as data:
            normalized_vars = {}
            for variable in ['omega', 'temp', 'qv']:
                values = data[variable].values
                normalized_vars[variable] = (values - min_val[variable]) / (max_val[variable] - min_val[variable])
            file_name = os.path.basename(file_path)
            normalized_data.append((file_name, normalized_vars))
    return normalized_data


def save_normalization_params(output_dir, min_val, max_val):
    """保存最小和最大值到指定目录。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for variable in min_val.keys():
        np.save(os.path.join(output_dir, f'min_{variable}.npy'), min_val[variable])
        np.save(os.path.join(output_dir, f'max_{variable}.npy'), max_val[variable])


def main():
    root_dir = '/project/caiman_datasets'
    output_dir = '/project/normalized_data'
    train_months = ['Jan2011', 'Feb2011', 'March2011', 'April2011', 'May2011']#'July2011', 'August2011', 'September2011'
    test_months = ['June2011']#'October2011'，'November2011', 'December2011'
 
   # 初始化归一化参数
    min_val = {'omega': np.inf, 'temp': np.inf, 'qv': np.inf}
    max_val = {'omega': -np.inf, 'temp': -np.inf, 'qv': -np.inf}
    # 计算训练数据的最小最大值
    for month in train_months:
        month_dir = os.path.join(root_dir, month)
        for file_path in glob.glob(os.path.join(month_dir, '*.nc')):
            with xr.open_dataset(file_path) as data:
                for variable in ['omega', 'temp', 'qv']:
                    values = data[variable].values
                    min_val[variable] = np.minimum(min_val[variable], np.min(values))
                    max_val[variable] = np.maximum(max_val[variable], np.max(values))

    # 保存归一化参数
    save_normalization_params(output_dir, min_val, max_val)

    # 不再计算归一化参数，直接加载
    min_val = {}
    max_val = {}
    for variable in ['omega', 'temp', 'qv']:
        min_val[variable] = np.load(os.path.join(output_dir, f'min_{variable}.npy'))
        max_val[variable] = np.load(os.path.join(output_dir, f'max_{variable}.npy'))

    train_omega, train_temperature, train_humidity = [], [], []
    test_omega, test_temperature, test_humidity = [], [], []

    # 分别加载和归一化训练和测试数据，并更新列表
    for month in tqdm(train_months, desc="Normalizing training data"):
        month_dir = os.path.join(root_dir, month)
        normalized_data = load_and_normalize_data(month_dir, min_val, max_val)
        for file_name, data in normalized_data:  # 正确地解包元组
            train_omega.append(data['omega'])  # 直接使用键访问
            train_temperature.append(data['temp'])
            train_humidity.append(data['qv'])

    for month in tqdm(test_months, desc="Normalizing test data"):
        month_dir = os.path.join(root_dir, month)
        normalized_data = load_and_normalize_data(month_dir, min_val, max_val)
        for file_name, data in normalized_data:
            test_omega.append(data['omega'])
            test_temperature.append(data['temp'])
            test_humidity.append(data['qv'])

    print("Training omega data shape:", np.array(train_omega).shape)
    print("Test temperature data shape:", np.array(test_temperature).shape)



    print("Data prepared and ready for training/testing.")
    # feature train
    train_and_get_hidden_state(train_omega, test_omega, 6, save_model_path='model_omega.pth', save_file_name='hidden_feature_omega.pt')
    train_and_get_hidden_state(train_temperature, test_temperature, 6, save_model_path='model_temperature.pth', save_file_name='hidden_feature_temperature.pt')
    train_and_get_hidden_state(train_humidity, test_humidity, 6, save_model_path='model_humidity.pth', save_file_name='hidden_feature_humidity.pt')
    
    # kmeans train
    train_k_means('hidden_feature_omega.pt', 'hidden_feature_temperature.pt', 'hidden_feature_humidity.pt')

    

if __name__ == "__main__":
    main()

    # predict()
   