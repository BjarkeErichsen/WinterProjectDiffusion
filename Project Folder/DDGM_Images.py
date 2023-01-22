import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from DataLoadingAndPrep import Digits
from FredeDataLoader import DataImage
from datetime import datetime
import datetime


#input eksperiment type
type_of_eksperiment = dict(using_conv = True)
using_conv = type_of_eksperiment['using_conv']
flatten = not using_conv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#normal hyperparams
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7
D = 64   # input dimension
M = 256  # the number of neurons in scale (s) and translation (t) nets
T = 7  #number of steps
beta = 0.4
s = 0.008    #Larger s -> Less curved beta curve
lr = 1e-4 #1e-4 # learning rate
num_epochs = 1 # max. number of epochs
max_patience = 2 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
batch_size = 32

#tilføjede hyperparametre
D = 13872
channels = 3
#networks:
if using_conv:
    conv_channels = 8
    k_size = 3 #must be 3, 5, 7 etc i e. not even numbers the first conv layer has K size + 2
    """
    p_dnns = [nn.Sequential(nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=k_size, padding = int((k_size-1)/2)), nn.ReLU(),
                            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding = int((k_size-1)/2)), nn.ReLU(),
                            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding=int((k_size-1)/2)), nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(512, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)).to(device=device) for _ in range(T-1)]
    decoder_net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=k_size, padding = int((k_size-1)/2)), nn.ReLU(),
                                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding = int((k_size-1)/2)), nn.ReLU(),
                                nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding=int((k_size-1)/2)), nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(512, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh()).to(device=device)
    """
    p_dnns = [nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),
        nn.Linear(16384, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2*D), nn.Tanh()).to(device=device) for _ in range(T-1)]

    decoder_net = nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),
        nn.Linear(16384, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, D), nn.Tanh()).to(device=device)
else:
    p_dnns = [nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)).to(device=device) for _ in range(T-1)]
    decoder_net = nn.Sequential(nn.Linear(D, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh()).to(device=device)
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


#helper functions
def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p
def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


class DDGM(nn.Module):
    def __init__(self, p_dnns, decoder_net, T, D, s):
        super(DDGM, self).__init__()

        print('DDGM by JT.')

        self.p_dnns = p_dnns  # a list of sequentials

        self.decoder_net = decoder_net

        # other params
        self.D = D

        self.T = T

        #self.beta = torch.FloatTensor([beta]).to(device=device)
        self.betas = cosine_beta_schedule(T, s=s).to(device=device)

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.betas[i]) * x + torch.sqrt(self.betas[i]) * torch.randn_like(x)

    def forward(self, x, reduction='avg'):
        # =====
        # forward difussion
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i)) #bemærk zs[-1] altså vi adder støj til sidste iteration

        # =====
        # backward diffusion
        mus = []
        log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1): #bemærk vi går i den negative retning 3, 2, 1, 0
            h = self.p_dnns[i](zs[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=-1)
            mus.append(mu_i)
            log_vars.append(log_var_i)


        mu_x = self.decoder_net(zs[0])
        if using_conv:
            _shape = x.shape
            x = x.reshape((_shape[0], _shape[1]*_shape[2]*_shape[3]))
            for i in range(len(zs)):
                zs[i] = zs[i].reshape((_shape[0], _shape[1]*_shape[2]*_shape[3]))

        # =====ELBO
        # RE        #loss for reconstruction of final layer p(X|Z)
        RE = log_standard_normal(x - mu_x).sum(-1)

        # KL        #the KL divergence of each layer ie. E[log(q(Z) / p(Z))]
        KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.betas[-1]) * zs[-1], torch.log(self.betas[-1])) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.betas[i]) * zs[i], torch.log(self.betas[i])) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1)
            KL = KL + KL_i


        # KL, RE
        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()

        return loss

    def sample(self, batch_size=64):
        z = torch.randn([batch_size, self.D]).to(device=device)
        if using_conv:
            z = torch.unsqueeze(z, 1)  # Bjarke added this
            z = z.reshape((z.shape[0], 3, 68, 68))
        for i in range(len(self.p_dnns) - 1, -1, -1):
            h = self.p_dnns[i](z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=-1) #splits the tensor into 2
            z = self.reparameterization(torch.tanh(mu_i), log_var_i)
            if using_conv:
                z = torch.unsqueeze(z, 1)  # Bjarke added this
                z = z.reshape((z.shape[0], 3, 68, 68))

        mu_x = self.decoder_net(z)

        return mu_x

    def sample_diffusion(self, x):
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        return zs[-1]

def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')
    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        #if using_conv:
            #test_batch = torch.unsqueeze(test_batch, 1) #Bjarke added this
        test_batch = test_batch.to(device=device)

        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 10**100
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            batch = batch.to(device = device)
            loss = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if indx_batch % 50 == 0:
                print(indx_batch)
            if indx_batch>200:
                break
        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('started')
            best_nll = loss_val
            torch.save(model, name + '.model')
        else:
            if loss_val < best_nll:
                torch.save(model, name + '.model')
                print('new best and saved')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val

#sample a real image
def samples_real(name, test_loader):
    # REAL-------
    num_x = 2
    num_y = 2
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.moveaxis(x[i], [0, 1, 2], [2, 0, 1])
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()
#sample from backward diffusion that are moved to results
def samples_generated(name, data_loader, extra_name=''):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 2
    num_y = 2
    x = model_best.sample(batch_size=num_x * num_y).cpu()

    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.moveaxis(x[i].reshape((3, 68, 68)), [0, 1, 2], [2, 0, 1])

        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()
#sample forward diffusion
def samples_diffusion(name, data_loader, extra_name=''):
    x = next(iter(data_loader))
    x = x.to(device=device)
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 2
    num_y = 2
    z = model_best.sample_diffusion(x)
    z = z.cpu()
    z = z.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.moveaxis(z[i].reshape((3, 68, 68)), [0, 1, 2], [2, 0, 1])
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        ax.imshow(plottable_image)
        ax.axis('off')

    plt.savefig(name + '_generated_diffusion' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()
#plot of nll = negative log likelihood, we of course want to minimize negative log likelihood
def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()

def final_test_and_saving(model, test_loader, nll_val):
    # Final evaluation
    test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    samples_real(result_dir + name, test_loader)
    plot_curve(result_dir + name, nll_val)

    # We generate a sample whenever we encounter a NEW BEST
    samples_generated(result_dir + name, test_loader, extra_name='FINAL')
    samples_diffusion(result_dir + name, test_loader, extra_name='DIFFUSION')

    sample_all_diffusion_steps(result_dir, name, test_loader)
    sample_all_backward_mapping_steps(result_dir, name)

def sample_all_diffusion_steps(result_dir, name, data_loader):
    x = next(iter(data_loader))[0]
    x = x.to(device=device)

    model_best = torch.load(result_dir + name + '.model')
    model_best.eval()

    dir = result_dir + name + "\ForwardDiffSteps"
    os.makedirs(dir)

    zs = [model_best.reparameterization_gaussian_diffusion(x, 0)]
    for i in range(1, model_best.T):
        zs.append(model_best.reparameterization_gaussian_diffusion(zs[-1], i))

    for i in range(T):
        z = zs[i].cpu()
        z = z.detach().numpy()
        plottable_image = np.moveaxis(z.reshape((3, 68, 68)), [0, 1, 2], [2, 0, 1])
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())

        plt.imshow(plottable_image, cmap='gray')
        plt.savefig(dir + '\Forward_Step' + str(i) + '.pdf')
        plt.close()

def sample_all_backward_mapping_steps(result_dir, name):
    # GENERATIONS-------
    model_best = torch.load(result_dir + name + '.model')
    model_best.eval()
    dir = result_dir + name + "\BackwardMapSteps"
    os.makedirs(dir)

    list_of_mu_i = []
    z = torch.randn([model_best.D]).to(device=device)
    if using_conv:
        z = z.reshape((1, 3, 68, 68))
    for i in range(len(model_best.p_dnns) - 1, -1, -1):
        h = model_best.p_dnns[i](z)
        mu_i, log_var_i = torch.chunk(h, 2, dim=-1)  # splits the tensor into 2
        list_of_mu_i.append(mu_i)
        z = model_best.reparameterization(torch.tanh(mu_i), log_var_i)
        if using_conv:
            z = z.reshape((1, 3, 68, 68))
    mu_x = model_best.decoder_net(z)
    list_of_mu_i.append(mu_x)

    for i in range(T):
        z = list_of_mu_i[i].cpu()
        z = z.detach().numpy()
        plottable_image = np.moveaxis(z.reshape((3, 68, 68)), [0, 1, 2], [2, 0, 1])
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        plt.imshow(plottable_image, cmap='gray')
        plt.savefig(dir + '\Backward_Step' + str(i) + '.pdf')
        plt.close()

if __name__ == "__main__":

    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)

    train_data = DataImage(mode='train', flatten = flatten)
    val_data = DataImage(mode='val', flatten = flatten)
    test_data = DataImage(mode='test', flatten = flatten)

    training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    now = datetime.datetime.now()
    name = 'Diffusion_CELL_IMAGES_' + "Hour_" + str(now.hour) + "_Min_" + str(now.minute) + '_' + "Conv_" + str(using_conv) + "_T_" + str(T) + '_' + "s" + str(s) + '_' + 'M_' + str(M)
    result_dir = 'Results/' + name + '/'
    if not (os.path.exists(result_dir)):
        os.makedirs(result_dir)

    model = DDGM(p_dnns, decoder_net, T=T, D=D, s=s)
    model = model.to(device=device)

    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model,
                       optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)

    final_test_and_saving(model, test_loader, nll_val)

