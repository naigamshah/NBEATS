#from google.colab import drive
#drive.mount('/content/drive/')

#!ls /content/drive/My\ Drive/Projects/Reflexis\ Colab/n-beats-master
path = "../Reflexis/QA/Uploads/Forecast/New_Data/"

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import os
import csv
from math import ceil

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn

import pandas as pd
import numpy as np

#from data import get_m4_data, dummy_data_generator
#from nbeats_pytorch.model import NBeatsNet # some import from the trainer script e.g. load/save functions.

CHECKPOINT_NAME = 'nbeats-training-chkpnt-GALLONS_SOLD_Blairs Bridge_gs=3000_fl=96.th'

# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
# It's a toy example to show how to do time series forecasting using N-Beats.

def get_m4_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = path+'examples/data/m4/train/Daily-train.csv'
    else:
        filename = path+'examples/data/m4/val/Daily-test.csv'

    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
                time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y

def dummy_data_generator(backcast_length, forecast_length, signal_type='seasonality', random=False, batch_size=32):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        if random:
            offset = np.random.standard_normal() * 0.1
        else:
            offset = 1
        if signal_type == 'trend':
            a = lin_space + offset
        elif signal_type == 'seasonality':
            a = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
            a += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
            a += lin_space * offset + np.random.rand() * 0.1
        elif signal_type == 'cos':
            a = np.cos(2 * np.pi * lin_space)
        else:
            raise Exception('Unknown signal type.')

        x = a[:backcast_length]
        y = a[backcast_length:]

        min_x, max_x = np.minimum(np.min(x), 0), np.max(np.abs(x))

        x -= min_x
        y -= min_x

        x /= max_x
        y /= max_x

        return x, y

    def gen():
        while True:
            xx = []
            yy = []
            for i in range(batch_size):
                x, y = get_x_y()
                xx.append(x)
                yy.append(y)
            yield np.array(xx), np.array(yy)

    return gen()

class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self, device, stack_types=(TREND_BLOCK, SEASONALITY_BLOCK), nb_blocks_per_stack=3, forecast_length=5, backcast_length=10, thetas_dims=(4, 8), share_weights_in_stack = False, hidden_layer_units=256, nb_harmonics=None):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.device = device
        #print(f'| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print('| --  Stack '+ str(stack_type.title()) + '(#' + str(stack_id) + ') (share_weights_in_stack=' + str(self.share_weights_in_stack) + ')')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],self.device, self.backcast_length, self.forecast_length, self.nb_harmonics)
                self.parameters.extend(block.parameters())
            print('     | -- ' + str(block))
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast

def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        #return '{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, share_thetas={self.share_thetas}) at @{id(self)}'
        return str(block_type) + "(units=" + str(self.units) + ", thetas_dim=" + str(self.thetas_dim) + ", backcast_length=" + str(self.backcast_length) + ", forecast_length=" + str(self.forecast_length) + ", share_thetas=" + str(self.share_thetas) + " at @" + str(id(self))

class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics= None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics= None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast

# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr

# trainer
def train_200_grad_steps(data, device, net, optimiser, test_losses):
	previous_loss = min(test_losses[-1],1000)
	global_step = load_temp(net, optimiser)
	patience = 100
	for x_train_batch, y_train_batch in data:
		global_step += 1
		optimiser.zero_grad()
		net.train()
		_, forecast = net(torch.tensor(x_train_batch, dtype=torch.float))
		loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float))
		loss.backward()
		optimiser.step()

		######
		net.eval()
		_, forecast = net(torch.tensor(x_val, dtype = torch.float))
		val_loss = F.mse_loss(forecast, torch.tensor(y_val, dtype=torch.float))
		if val_loss <= previous_loss:
			with torch.no_grad():
				save_temp(net,optimiser, global_step, "temp_chk_pnt.th")
			previous_loss = val_loss
			patience = 100
		else:
			patience -= 1
			if patience == 0:
				return 0
		######

		if global_step % 30 == 0:
			print('grad_step = ' + str(str(global_step).zfill(6)) + ', tr_loss = ' + str(loss.item()) + ', te_loss = ' + str(val_loss.item()))
		if global_step > 0 and global_step % 200 == 0:
			with torch.no_grad():
				save(net, optimiser, global_step)
			break

# loader/saver for checkpoints.
def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print('Restored checkpoint from ' + str(CHECKPOINT_NAME) + '.')
        return grad_step
    return 0

def load_temp(model, optimiser):
	temp_checkpoint_name = "GS_BB_3_0.009668644517660141.th"
	if os.path.exists(temp_checkpoint_name):
		checkpoint = torch.load(temp_checkpoint_name)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
		grad_step = checkpoint['grad_step']
		print('Restored checkpoint from ' + str(temp_checkpoint_name) + '.')
		return grad_step
	return 0

def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)

def save_temp(model, optimiser, grad_step, checkpoint_name):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, checkpoint_name)

# evaluate model on test data and produce some plots.
def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        #preds = pd.DataFrame(data = ff , columns = ['y_hat'])
        #preds['y'] = yy
        #preds.to_csv(path + "predictions_GALLONS_SOLD_Blairs Bridge_gs=3000_fl=96.csv", index = False)
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    #plt.show()
    plt.savefig('plot1_gs=3000_fl=96.png')

# main
if os.path.isfile(CHECKPOINT_NAME):
    os.remove(CHECKPOINT_NAME)
device = torch.device('cpu')  # use the trainer.py to run on GPU.
#forecast_length = 96
#backcast_length = 7 * forecast_length
batch_size = 32  # greater than 4 for viz

milk = pd.read_csv(path+'new_data_GALLONS_SOLD_Blairs Bridge.csv', index_col=0, parse_dates=True)

print(milk.head())
milk = milk.values  # just keep np array here for simplicity.
norm_constant = np.max(milk)
print(norm_constant)
milk = milk / norm_constant  # small leak to the test set here.
#print(milk)
"""
x_train_batch, y = [], []
for i in range(backcast_length, len(milk) - forecast_length):
    x_train_batch.append(milk[i - backcast_length:i])
    y.append(milk[i:i + forecast_length])

x_train_batch = np.array(x_train_batch)[..., 0]
y = np.array(y)[..., 0]
print(len(x_train_batch)/96.0)
c = int(len(x_train_batch) * 0.7)
d = int(len(x_train_batch) * 0.8)
x_train, y_train = x_train_batch[:c], y[:c]
x_val, y_val = x_train_batch[c:d], y[c:d]
x_test, y_test = x_train_batch[d:], y[d:]

print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)
"""
min_loss = 1000
hiddenLayerUnits = [64, 128, 256]
stackTypes = [[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK]]
numBlocksPerStack = [4]
backcastDays = [7, 14, 21]

backcast_length = 7 * 96
forecast_length = 96
#x_train_batch, y = [], []
x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []

train_length = ceil((len(milk)/96) * 0.70) * 96
val_length = ceil((len(milk)/96) * 0.15) * 96
test_length = len(milk) - train_length - val_length
print("Lengths:\nFull: " + str(len(milk)) + " Train: " + str(train_length) + " Val: " + str(val_length) + " Test: " + str(test_length))

#for i in range(backcast_length, len(milk) - forecast_length):
    #x_train_batch.append(milk[i - backcast_length:i])
    #y.append(milk[i:i + forecast_length])

for i in range(backcast_length, train_length - forecast_length):
    x_train.append(milk[i - backcast_length:i])
    y_train.append(milk[i:i + forecast_length])

for i in range(train_length + backcast_length, train_length + val_length - forecast_length, 96):
    x_val.append(milk[i - backcast_length:i])
    y_val.append(milk[i:i + forecast_length])

for i in range(train_length + val_length + backcast_length, len(milk) - forecast_length, 96):
    x_test.append(milk[i - backcast_length:i])
    y_test.append(milk[i:i + forecast_length])

x_train, y_train = np.array(x_train)[...,0], np.array(y_train)[...,0]
x_val, y_val = np.array(x_val)[...,0], np.array(y_val)[...,0]
x_test, y_test = np.array(x_test)[...,0], np.array(y_test)[...,0]

print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

net = NBeatsNet(device = device, stack_types=stackTypes[0], nb_blocks_per_stack=numBlocksPerStack[0], forecast_length=forecast_length, backcast_length=backcast_length, thetas_dims=[7,8], share_weights_in_stack=False, hidden_layer_units=hiddenLayerUnits[0])
optimiser = optim.Adam(net.parameters())

load_temp(net,optimiser)

net.eval()
_, forecast = net(torch.tensor(x_test, dtype = torch.float))
test_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype = torch.float)).item()
print(test_loss)
p = forecast.detach().numpy()
print(len(p))
for i in range(len(x_test)):
    print("Test series #" + str(i))
    ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
    preds = pd.DataFrame(data = yy, columns = ['y'])
    preds['yhat'] = ff
    preds.to_csv("ErrorBars/GS_BB/3/pred" + str(i) + ".csv", index = False)
"""
for i in range(len(x_test)):
    #net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        preds = pd.DataFrame(data = ff , columns = ['y_hat'])
        preds['y'] = yy
        preds.to_csv(path + "predictions_GALLONS_SOLD_Blairs Bridge_gs=3000_fl=96.csv", index = False)
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    #plt.show()
    plt.savefig('plot1_gs=3000_fl=96.png')


for days in backcastDays:
    backcast_length = days * 96
    forecast_length = 96
    #x_train_batch, y = [], []
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    train_length = ceil((len(milk)/96) * 0.75) * 96
    val_length = ceil((len(milk)/96) * 0.15) * 96
    test_length = len(milk) - train_length - val_length
    print("Lengths:\nFull: " + str(len(milk)) + " Train: " + str(train_length) + " Val: " + str(val_length) + " Test: " + str(test_length))

    #for i in range(backcast_length, len(milk) - forecast_length):
        #x_train_batch.append(milk[i - backcast_length:i])
        #y.append(milk[i:i + forecast_length])

    for i in range(backcast_length, train_length - forecast_length):
        x_train.append(milk[i - backcast_length:i])
        y_train.append(milk[i:i + forecast_length])

    for i in range(train_length + backcast_length, train_length + val_length - forecast_length, 96):
        x_val.append(milk[i - backcast_length:i])
        y_val.append(milk[i:i + forecast_length])

    for i in range(train_length + val_length + backcast_length, len(milk) - forecast_length, 96):
        x_test.append(milk[i - backcast_length:i])
        y_test.append(milk[i:i + forecast_length])

    x_train, y_train = np.array(x_train)[...,0], np.array(y_train)[...,0]
    x_val, y_val = np.array(x_val)[...,0], np.array(y_val)[...,0]
    x_test, y_test = np.array(x_test)[...,0], np.array(y_test)[...,0]

    #x_train_batch = np.array(x_train_batch)[..., 0]
    #y = np.array(y)[..., 0]
    #print(len(x_train_batch)/96.0)
    #c = int(len(x_train_batch) * 0.7)
    #d = int(len(x_train_batch) * 0.8)
    #x_train, y_train = x_train_batch[:c], y[:c]
    #x_val, y_val = x_train_batch[c:d], y[c:d]
    #x_test, y_test = x_train_batch[d:], y[d:]

    print(x_train.shape, x_test.shape, x_val.shape)
    print(y_train.shape, y_test.shape, y_val.shape)
   
    for a in range(len(stackTypes)):
	    for b in range(len(numBlocksPerStack)):
		    for c in range(len(hiddenLayerUnits)):
                            print("Iteration: " + str(days) + " " + str(a) + " " + str(b) + " " + str(c))

                            # model
                            if a==0:
                                thetadims = [7,8]
                            else:
                                thetadims = [2,8]

                            net = NBeatsNet(device = device, stack_types=stackTypes[a], nb_blocks_per_stack=numBlocksPerStack[b], forecast_length=forecast_length, backcast_length=backcast_length, thetas_dims=thetadims, share_weights_in_stack=False, hidden_layer_units=hiddenLayerUnits[c])
                            optimiser = optim.Adam(net.parameters())
                            save_temp(net,optimiser,0,"temp_chk_pnt.th")

                            # data
                            data = data_generator(x_train, y_train, batch_size)

			    # training
			    # model seems to converge well around ~2500 grad steps and starts to overfit a bit after.
                            test_losses = []
                            for i in range(100):
                                    eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_val, y_val)
                                    flag = train_200_grad_steps(data, device, net, optimiser, test_losses)

                                    if flag==0:
                                            global_step = load_temp(net,optimiser)
                                            save_temp(net ,optimiser, global_step, "model[" + str(days) + "]_" + str(a) + "_" + str(b) + "_" + str(c) + "_" + str(test_losses[-1]) + ".th")
                                            print("Model saved as: model[" + str(days) + "]_" + str(a) + "_" + str(b) + "_" + str(c) + "_" + str(test_losses[-1]) + ".th")
                                            if test_losses[-1] < min_loss:
                                                    min_loss = test_losses[-1]
                                                    opt_a = a
                                                    opt_b = b
                                                    opt_c = c
                                                    print("Optimal Model till now!")
                                            print("Stopped Early!")
                                            break
"""
