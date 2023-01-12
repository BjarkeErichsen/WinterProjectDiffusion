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
import wandb

#input eksperiment type
type_of_eksperiment = dict(using_conv = False, Using_image_dataset = False, run_sweep = False)
using_conv = type_of_eksperiment['using_conv']
Using_image_dataset = type_of_eksperiment['Using_image_dataset']
run_sweep = type_of_eksperiment['run_sweep']

#normal hyperparams
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7
D = 64   # input dimension
M = 256  # the number of neurons in scale (s) and translation (t) nets
T = 5  #number of steps
beta = 0.8
lr = 1e-3 #1e-4 # learning rate
num_epochs = 10 # max. number of epochs
max_patience = 10 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

#tilføjede hyperparametre
if Using_image_dataset:
    D = 13872
conv_channels = 8
batch_size = 32

#networks:
if using_conv:
    p_dnns = [nn.Sequential(nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding = 1), nn.ReLU(),
                            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding = 1), nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(512, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)) for _ in range(T-1)]
    decoder_net = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding = 1), nn.ReLU(),
                                nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding = 1), nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(512, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh())
else:

    p_dnns = [nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)) for _ in range(T-1)]
    decoder_net = nn.Sequential(nn.Linear(D, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh())


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
    def __init__(self, p_dnns, decoder_net, beta, T, D):
        super(DDGM, self).__init__()

        print('DDGM by JT.')

        self.p_dnns = p_dnns  # a list of sequentials

        self.decoder_net = decoder_net

        # other params
        self.D = D

        self.T = T

        self.beta = torch.FloatTensor([beta])

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)

    def forward(self, x, reduction='avg'):
        # =====
        # forward difussion
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        # =====
        # backward diffusion
        mus = []
        log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):
            h = self.p_dnns[i](zs[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            mus.append(mu_i)
            log_vars.append(log_var_i)

        mu_x = self.decoder_net(zs[0])

        # =====ELBO
        # RE
        RE = log_standard_normal(x - mu_x).sum(-1)

        # KL
        KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-1], torch.log(self.beta)) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.beta) * zs[i], torch.log(self.beta)) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1)

            KL = KL + KL_i
        # KL, RE
        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()

        return loss

    def sample(self, batch_size=64):
        z = torch.randn([batch_size, self.D])
        if using_conv:
            z = torch.unsqueeze(z, 1)  # Bjarke added this

        for i in range(len(self.p_dnns) - 1, -1, -1):
            h = self.p_dnns[i](z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=1) #splits the tensor into 2
            z = self.reparameterization(torch.tanh(mu_i), log_var_i)
            if using_conv:
                z = torch.unsqueeze(z, 1)  # Bjarke added this

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
        if using_conv:
            test_batch = torch.unsqueeze(test_batch, 1) #Bjarke added this

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
    best_nll = 1000.
    patience = 0


    wandb.watch(model, log="all", log_freq = 1000)
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            if using_conv:
                batch = torch.unsqueeze(batch, 1)  # Bjarke added this

            loss = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting
        wandb.log({'validation nll':loss_val }, step=e)
        if e == 0:
            torch.save(model, name + '.model')
            print('started')
            best_nll = loss_val
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
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()
#sample from backward diffusion that are moved to results
def samples_generated(name, data_loader, extra_name=''):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()
#sample forward diffusion
def samples_diffusion(name, data_loader, extra_name=''):
    x = next(iter(data_loader))

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    z = model_best.sample_diffusion(x)
    z = z.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(z[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
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

def test(model, test_loader, nll_val):
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


    #the following is just a convoluted way of saving the model
    model.eval()
    with torch.no_grad():
        for indx_batch, test_batch in enumerate(test_loader):
            if using_conv:
                test_batch = torch.unsqueeze(test_batch, 1) #Bjarke added this
            dummy_input = test_batch
            outputs = model(test_batch)
            break
    torch.onnx.export(model, dummy_input, "model.onnx", input_names = "noice", output_names = "digit")
    wandb.save("model.onnx")





def make(config):
    T = config.T
    M = config.M
    model = DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D)
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    return model, optimizer
def model_pipeline(hyperparams):

    with wandb.init(project="pytorch-demo", config = hyperparams):
        config = wandb.config
        # Eventually, we initialize the full model
        model, optimizer = make(config)
        print(model)

        # Training procedure
        nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                               training_loader=training_loader, val_loader=val_loader)

        #test(model, test_loader, nll_val)



if __name__ == "__main__":

    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)
    if Using_image_dataset:
        train_data = DataImage(mode='train')
        val_data = DataImage(mode='val')
        test_data = DataImage(mode='test')
    else:
        train_data = Digits(mode='train', transforms=transforms)
        val_data = Digits(mode='val', transforms=transforms)
        test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    name = 'Diffusion' + '_' + "T_" + str(T) + '_' + "beta_" + str(beta) + '_' + 'M_' + str(M)
    result_dir = 'Results/' + name + '/'
    if not (os.path.exists(result_dir)):
        os.makedirs(result_dir)



    if run_sweep:
        sweep_config = {
            'method': 'grid',
            'name': 'sweep',
            'metric': {
                'name': 'nll',
                'goal': 'minimize'},
            'parameters': {
                'T': {'values': [3, 6, 15]},
                'M': {'values': [256, 64]}
            }
        }
            #.init(project="Diffusion", config = sweep_config)
            #sweep_id = wandb.sweep(sweep=sweep_config, project="Diffusion")
            #wandb.agent(sweep_id, function=model_pipeline, count = 3)
    else:
        config = dict(T = 6, M = 65)
        model_pipeline(config)

    model = DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D)
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model,
                       optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)
    test(model, test_loader, nll_val)


