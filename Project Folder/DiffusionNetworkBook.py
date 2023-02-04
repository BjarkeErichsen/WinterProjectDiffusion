import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as tt
from DataLoadingAndPrep import Digits
from FredeDataLoader import DataImage
import datetime
import seaborn as sns

#input eksperiment type
type_of_eksperiment = dict(using_conv = False)
using_conv = type_of_eksperiment['using_conv']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collect_data = False

#normal hyperparams
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7
D = 64   # input dimension
M = 256  # the number of neurons in scale (s) and translation (t) nets
T = 20  #number of steps
beta = 0.05
s = 0.003  #pertains to the cosine noice scheduler. The lower the more parabolic the curve is
lr = 1e-5 # learning rate
num_epochs = 150 # max. number of epochs
max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
batch_size = 32

#tilføjede hyperparametre



#networks:
if using_conv:
    conv_channels = 8
    k_size = 3 #must be 3, 5, 7 etc i e. not even number

    p_dnns = [nn.Sequential(nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=k_size, padding = int((k_size-1)/2)), nn.ReLU(),
                            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding = int((k_size-1)/2)), nn.ReLU(),
                            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding=int((k_size-1)/2)), nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(512, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D), nn.Tanh()).to(device=device) for _ in range(T-1)]
    decoder_net = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=k_size, padding = int((k_size-1)/2)), nn.ReLU(),
                                nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding = int((k_size-1)/2)), nn.ReLU(),
                                nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=k_size,padding=int((k_size-1)/2)), nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(512, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh()).to(device=device)
else:
    p_dnns = [nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Dropout(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Dropout(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Dropout(),
                            nn.Linear(M, 2 * D)).to(device=device) for _ in range(T-1)]

    decoder_net = nn.Sequential(nn.Linear(D, M*2), nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(M*2, D), nn.Tanh()).to(device=device)


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

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.tensor([beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta])
    return torch.clip(betas, 0.0001, 0.9999)



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

        self.batch_data = None
    #zt <- mu and var <- network(zt+1)
    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    #def reparameterization_gaussian_diffusion(self, x, i):
    #    return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.betas[i]) * x + torch.sqrt(self.betas[i]) * torch.randn_like(x)
        #return torch.sqrt(1. - self.betas[i]) * x + self.betas[i] * torch.randn_like(x)

    def forward(self, x, reduction='avg', collect_data = False):
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

        mu_x = self.decoder_net(zs[0].to(device=device))

        # =====ELBO
        RE = log_standard_normal(x - mu_x).sum(-1)

        if collect_data:
            plot_dict = {"RE": [], "qz0": [], "pz0": [], "qz": [], "pz": [], "mus": [], "log_var": []}
            qz = log_normal_diag(zs[-1], torch.sqrt(1. - self.betas[-1]) * zs[-1], torch.log(self.betas[-1]))
            pz = log_standard_normal(zs[-1])
            KL = (qz - pz).sum(-1)
            plot_dict["qz0"].append(torch.mean(qz.sum(-1)))
            plot_dict["pz0"].append(torch.mean(pz.sum(-1)))
            plot_dict["RE"].append(torch.mean(pz.sum(-1)))
        else:
            KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.betas[-1]) * zs[-1], torch.log(self.betas[-1])) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):

            if collect_data:
                qz = log_normal_diag(zs[i], torch.sqrt(1. - self.betas[i]) * zs[i], torch.log(self.betas[i]))
                pz = log_normal_diag(zs[i], mus[i], log_vars[i])
                KL_i = (qz - pz).sum(-1)
                plot_dict["qz"].append(torch.mean(qz.sum(-1)))
                plot_dict["pz"].append(torch.mean(pz.sum(-1)))
                plot_dict["mus"].append(mus[i])
                plot_dict["log_var"].append(log_vars[i])
                self.batch_data = plot_dict
            else:
                KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.betas[i]) * zs[i], torch.log(self.betas[i])) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1)

            KL = KL + KL_i

        #min negative nll,

        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()

        return loss

    # sample bakward diffusion from random start
    def sample(self, batch_size=64):
        z = torch.randn([batch_size, self.D]).to(device=device)
        if using_conv:
            z = torch.unsqueeze(z, 1)  # Bjarke added this

        for i in range(len(self.p_dnns) - 1, -1, -1):
            h = self.p_dnns[i](z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=1) #splits the tensor into 2
            z = self.reparameterization(mu_i, log_var_i)
            if using_conv:
                z = torch.unsqueeze(z, 1)  # Bjarke added this

        mu_x = self.decoder_net(z)

        return mu_x

    #forward diffusion
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
    #batch_data = []
    for indx_batch, test_batch in enumerate(test_loader):
        if using_conv:
            test_batch = torch.unsqueeze(test_batch, 1) #Bjarke added this
        test_batch = test_batch.to(device=device)

        loss_t = model_best.forward(test_batch, reduction='sum')
        #batch_data.append(model_best.batch_data)
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N
    #plot_epoch_data(batch_data)
    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    model = model.cuda()

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        if collect_data:
            epoch_data = []
        for indx_batch, batch in enumerate(training_loader):
            if using_conv:
                batch = torch.unsqueeze(batch, 1)  # Bjarke added this
            batch = batch.to(device=device)
            loss = model.forward(batch, collect_data=collect_data)
            if collect_data:
                epoch_data.append(model.batch_data)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if collect_data:
            plot_epoch_data(epoch_data)
        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
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
    x = model_best.sample(batch_size=num_x * num_y).cpu()
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
    x = next(iter(data_loader)).to(device=device)

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4

    z = model_best.sample_diffusion(x)
    z = z.cpu()
    z = z.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(z[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()
#plot of nll = negative log likelihood, we of course want to minimize negative log likelihood
def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()

def plot_epoch_data(epoch_data):

    plot_dict = {"RE": [], "qz0": [], "pz0": [], "qz": [], "pz": [], "mus": [], "log_var": []}
    n_batches = len(epoch_data)

    qz0 = []
    pz0 = []
    qz = []
    pz = []
    RE = []
    mus = []
    log_var = []
    for i in range(n_batches):
        qz0.append(epoch_data[i]["qz0"])
        pz0.append(epoch_data[i]["pz0"])
        qz.append(epoch_data[i]["qz"])
        pz.append(epoch_data[i]["pz"])
        RE.append(epoch_data[i]["RE"])
        mus.append(epoch_data[i]["mus"])
        log_var.append(epoch_data[i]["log_var"])

    plt.plot(range(n_batches), qz0)
    plt.plot(range(n_batches), pz0)
    plt.title('pz and qz at timestep 0, for every batch')
    plt.show()


    new_pz = []
    new_qz = []
    for i in range(T-1):
        helper1 = []
        helper2 = []
        for k in range(n_batches):
            helper1.append(pz[k][i])
            helper2.append(qz[k][i])
        new_pz.append(helper1)
        new_qz.append(helper2)
    for i in range(T-1):
        plt.plot(range(n_batches), new_pz[i])
    plt.title('pz for each diffusion step over each batch')
    plt.show()

    for i in range(T - 1):
        plt.plot(range(n_batches), new_qz[i])
    plt.title('qz for each diffusion step over each batch')
    plt.show()

    plt.plot(range(n_batches), RE)
    plt.title('RE')
    plt.show()



    rTimestep1 = np.random.randint(0, T-2)
    rBatch1 = np.random.randint(0, n_batches-1)
    rBatchElement1 = np.random.randint(0, batch_size-1)
    rTimestep2 = np.random.randint(0, T-2)
    rBatch2 = np.random.randint(0, n_batches-1)
    rBatchElement2 = np.random.randint(0, batch_size-1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #fig.suptitle('Sharing x per column, y per row')
    sns.heatmap(mus[rBatch1][rTimestep1][rBatchElement1].detach().cpu().numpy().reshape(8, 8), linewidth=0.5, cmap='coolwarm', ax=ax1)
    sns.heatmap(mus[rBatch2][rTimestep2][rBatchElement2].detach().cpu().numpy().reshape(8, 8), linewidth=0.5, cmap='coolwarm', ax=ax2)
    sns.heatmap(torch.exp(log_var[rBatch1][rTimestep1][rBatchElement1]).detach().cpu().numpy().reshape(8, 8), linewidth=0.5, ax=ax3)
    sns.heatmap(torch.exp(log_var[rBatch2][rTimestep2][rBatchElement2]).detach().cpu().numpy().reshape(8, 8), linewidth=0.5, ax=ax4)
    """
    ax1.title.set_text('Means at Timestep ' + str(rTimestep1))
    ax2.title.set_text('Means at Timestep ' + str(rTimestep2))
    ax3.title.set_text('Variance at Timestep ' + str(rTimestep1))
    ax4.title.set_text('Variance  at Timestep ' + str(rTimestep2))
    """
    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()

    KL = np.squeeze(np.array(qz0)-np.array(pz0))
    for i in range(T-1):
        KL += np.array(new_qz[i])-np.array(new_pz[i])
    plt.plot(range(n_batches),  KL, label='KL')
    plt.plot(range(n_batches), RE, label='RE')
    plt.legend()
    plt.title("All the different measures of loss")
    plt.xlabel("Epochs")
    plt.show()

    plt.plot(range(n_batches), RE, label='RE')
    plt.title("Reconstruction loss")
    plt.show()
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

    plt.plot(cosine_beta_schedule(T, s=s), label = str(s))
    plt.savefig(result_dir + name + 'NoiceScheduler.pdf', bbox_inches='tight')
    plt.close()

    sample_all_diffusion_steps(result_dir, name, test_loader)
    sample_all_backward_mapping_steps(result_dir, name)

def sample_all_diffusion_steps(result_dir, name, data_loader):
    x = next(iter(data_loader))[0]
    x = x.to(device=device)

    model_best = torch.load(result_dir + name + '.model')
    model_best.eval()

    dir = result_dir + name + "\ForwardDiffSteps"
    os.makedirs(dir)
    zs = [x]
    zs.append(model_best.reparameterization_gaussian_diffusion(x, 0))
    for i in range(1, model_best.T):
        zs.append(model_best.reparameterization_gaussian_diffusion(zs[-1], i))

    for i in range(T+1):
        z = zs[i].cpu()
        z = z.detach().numpy()
        plottable_image = z.reshape((8, 8))
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        plt.imshow(plottable_image, cmap='gray')
        if i == 0:
            plt.savefig(dir + '\X_OriginalData' + '.pdf')
            plt.close()
        else:
            plt.savefig(dir + '\Forward_Step' + str(i - 1) + '.pdf')
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
        z = z.reshape((1, 8, 8))

    list_of_mu_i.append(z)
    for i in range(len(model_best.p_dnns) - 1, -1, -1):
        h = model_best.p_dnns[i](z)
        mu_i, log_var_i = torch.chunk(h, 2, dim=-1)  # splits the tensor into 2
        list_of_mu_i.append(mu_i)
        z = model_best.reparameterization(torch.tanh(mu_i), log_var_i)
        if using_conv:
            z = z.reshape((1, 8, 8))
    mu_x = model_best.decoder_net(z)
    list_of_mu_i.append(mu_x)

    for i in range(T+1):
        z = list_of_mu_i[i].cpu()
        z = z.detach().numpy()
        plottable_image = z.reshape((8, 8))
        plottable_image = (plottable_image - plottable_image.min()) / (plottable_image.max() - plottable_image.min())
        plt.imshow(plottable_image, cmap='gray')
        if i == 0:
            plt.savefig(dir + '\zT_pureNoice''.pdf')
            plt.close()
        else:
            plt.savefig(dir + '\Backward_Step' + str(i-1) + '.pdf')
            plt.close()



if __name__ == "__main__":

    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)

    train_data = Digits(mode='train', transforms=transforms)
    val_data = Digits(mode='val', transforms=transforms)
    test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    now = datetime.datetime.now()
    name = 'Diffusion' + '_'  + "Hour_" + str(now.hour) + "_Min_" + str(now.minute) + '_' + "Conv_" + str(using_conv) + "_T_" + str(T) + '_' + "s_" + str(s) + '_' + 'M_' + str(M)
    result_dir = 'Results/' + name + '/'
    if not (os.path.exists(result_dir)):
        os.makedirs(result_dir)

    model = DDGM(p_dnns, decoder_net, s = s, T=T, D=D)

    model = model.to(device=device)

    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model,
                       optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)
    test(model, test_loader, nll_val)



