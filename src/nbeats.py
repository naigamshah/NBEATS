#Author: Naigam Shah
#For: Reflexis Team
##Import Libraries  
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
import argparse
import pickle

##Argument Parser

# General arguments
parser = argparse.ArgumentParser(description ='Search some files')   
parser.add_argument('-d', '--data', required = True, dest = 'input_file_path', action = 'store', help = 'Specify input datafile path')
parser.add_argument('-o', '--out', required = True, dest = 'output_directory', action = 'store', help = 'Specify output data storage directory')
parser.add_argument('-chk', '--check-point', required = False, dest = 'checkpoint_name', action = 'store', help = 'Specify checkpoint name', default = "checkpoint.th")
parser.add_argument('-sd', '--show-development', dest = 'show_development', action = 'store_true', help = 'Activate flag to show mid-stage development of model')
parser.add_argument('-nc', '--norm-constant', required = False, dest = 'norm_constant', action = 'store', help = 'Specify normalizing constant', type = float)
parser.add_argument('-md', '--model-dictionary', required = True, dest = 'dict_path', action = 'store', help = 'Specify model dictionary pickle file path', type = str)

# Model building arguments (multiple can be provided for each except model and forecast_days)
parser.add_argument('-bc', '--backcast-days', required = False, dest = 'backcast_days', action = 'store', help = 'Specify backcast length in days', nargs = '*', type = int, default = 7)          
parser.add_argument('-fc', '--forecast-days', required = False, dest = 'forecast_days', action = 'store', help = 'Specify forecast length in days', type = int, default = 1)
parser.add_argument('-hlu', '--hidden-layer-units', required = False, dest = 'hidden_layer_units', action = 'store', help = 'Specify number of units in hidden layer', nargs = '*', type = int, default = 64)
parser.add_argument('-bps', '--blocks-per-stack', required = False, dest = 'blocks_per_stack', action = 'store', help = 'Specify number of blocks per stack', nargs = '*', type = int, default = 4)
parser.add_argument('-sw', '--share-weights', dest = 'share_weights_in_stack', action = 'store_true', help = 'Activate flag for the model to share weights in stack')
parser.add_argument('-st', '--stack-type', required = False, dest = 'stack_type', action = 'append', help = 'Specify stack types in any combination of generic, seasonality or trend', nargs = '*', type = str, choices = ['generic','seasonality','trend'], default=[])
parser.add_argument('-td', '--theta-dims', required = False, dest = 'theta_dims', action = 'append', help = 'Specify theta dimensions', nargs = '*', type = int, default = [])

# Training phase arguments
parser.add_argument('-mp', '--model-prefix', required = False, dest = 'model_prefix', action = 'store', help = 'Specify entire path of model (with model prefix)', default="model")
parser.add_argument('-val', '--validation-ratio', required = False, dest = 'validation_ratio', action = 'store', help = 'Specify validation ratio for validation during training', type = float, default = 0.2)
parser.add_argument('-pp', '--patience-period', required = False, dest = 'patience_period', action = 'store', help = 'Specify patience period for Early Stopping', type = int, default = 100)
parser.add_argument('-bs', '--batch-size', required = False, dest = 'batch_size', action = 'store', help = 'Specify batch size while training', type = int, default = 32)

# Testing phase arguments
parser.add_argument('-m', '--model', required = False, dest = 'model', action = 'store', help = 'Specify entire path of model (with model name)')
parser.add_argument('-t', '--test', dest = 'testing', action = 'store_true', help = 'Activate flag in testing phase. Input data path becomes test file')
parser.add_argument('-cp', '--comparison-plots', dest = 'comparison_plots', action = 'store_true', help = 'Activate flag to store comparison plots in given output directory')

# Assigning arguments to corresponding variables
args = parser.parse_args() 
INPUT_FILE_PATH = args.input_file_path
OUTPUT_DIRECTORY = args.output_directory
CHECKPOINT_NAME = args.checkpoint_name
SHOW_DEVELOPMENT = args.show_development
DICT_PATH = args.dict_path
MODEL_PREFIX = args.model_prefix
BACKCAST_DAYS = args.backcast_days 
FORECAST_DAYS = args.forecast_days
HIDDEN_LAYER_UNITS = args.hidden_layer_units
BLOCKS_PER_STACK = args.blocks_per_stack
SHARE_WEIGHTS_IN_STACK = args.share_weights_in_stack
if len(args.theta_dims)==0: args.theta_dims.append([7,8])
THETA_DIMS = args.theta_dims
if len(args.stack_type)==0: args.stack_type.append(['generic','generic'])
VALIDATION_RATIO = args.validation_ratio
PATIENCE_PERIOD  = args.patience_period
BATCH_SIZE = args.batch_size
MODEL = args.model
TESTING = args.testing
COMPARISON_PLOTS = args.comparison_plots

if TESTING:
    assert MODEL!=None, "Full model name with path should be specified while testing."
assert INPUT_FILE_PATH[-4:]==".csv", "Your input file should be of .csv format"
assert VALIDATION_RATIO < 1.0 , "Please keep validation ratio to be less than 1.0"
assert len(args.theta_dims) == len(args.stack_type), "Please make sure your theta_dims and stack_type has equal number of entries."
for i in range(len(args.stack_type)):
    assert len(args.theta_dims[i])== len(args.stack_type[i]), "Specified " + str(i) + "th entry dimensions of theta_dims and stack_type are not matching."
OUTPUT_DIRECTORY = OUTPUT_DIRECTORY if OUTPUT_DIRECTORY[-1]=='/' else OUTPUT_DIRECTORY+'/'
DICT_PATH = DICT_PATH if DICT_PATH[-7:]==".pickle" else DICT_PATH+".pickle"
if COMPARISON_PLOTS and os.path.isdir(OUTPUT_DIRECTORY + "Comparison_Plots/")==False:
    path = os.path.join(OUTPUT_DIRECTORY, "Comparison_Plots")
    os.mkdir(path)
    print("New directory created named 'Comparison_Plots' inside specified output directory.")
print("\nRead input file path: " + str(os.path.abspath(INPUT_FILE_PATH)))
print("Read output directory: " + str(os.path.abspath(OUTPUT_DIRECTORY)))


###################################################################################################
######################### NBeats main code starts #################################################
###################################################################################################

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

##NBeats Model Class is defined here
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
        print('| N-Beats')
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

##Different kinds of models are listed down below
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

##One Block of Model
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

##Different kinds of Blocks are listed down below
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

# plot utils. For plotting time series graphs
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
###################################################################################################


# trainer. Training model for 200 steps for now
def train_200_grad_steps(data, device, net, optimiser, test_losses):
	previous_loss = min(test_losses[-1],1000)
	global_step = load(net, optimiser, CHECKPOINT_NAME)
	patience = PATIENCE_PERIOD
	for x_train_batch, y_train_batch in data:
		global_step += 1
		optimiser.zero_grad()
		net.train()
		_, forecast = net(torch.tensor(x_train_batch, dtype=torch.float))
		loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float))
		loss.backward()
		optimiser.step()

		###### Early stopping implementation ######
		net.eval()
		_, forecast = net(torch.tensor(x_val, dtype = torch.float))
		val_loss = F.mse_loss(forecast, torch.tensor(y_val, dtype=torch.float))
		if val_loss <= previous_loss:
			with torch.no_grad():
				save(net,optimiser, global_step, CHECKPOINT_NAME)
			previous_loss = val_loss
			patience = PATIENCE_PERIOD
		else:
			patience -= 1
			if patience == 0:
				return 0
		######

		if global_step % 30 == 0:
			print('grad_step = ' + str(str(global_step).zfill(6)) + ', tr_loss = ' + str(loss.item()) + ', te_loss = ' + str(val_loss.item()))
		if global_step > 0 and global_step % 200 == 0:
			with torch.no_grad():
				save(net, optimiser, global_step,CHECKPOINT_NAME)
			break

# loader/saver for checkpoints.
def load(model, optimiser, checkpoint_name):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print('Restored checkpoint from ' + str(checkpoint_name) + '.')
        return grad_step
    return 0

def save(model, optimiser, grad_step, checkpoint_name):
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
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    if SHOW_DEVELOPMENT: 
    		plt.show()

# main
STACK_TYPE = []
for i in range(len(args.stack_type)):
    ST = []
    for j in range(len(args.stack_type[i])):
        if args.stack_type[i][j]=="generic":
            ST.append(NBeatsNet.GENERIC_BLOCK)
        elif args.stack_type[i][j]=="seasonality":
            ST.append(NBeatsNet.SEASONALITY_BLOCK)
        else:
            ST.append(NBeatsNet.TREND_BLOCK)
    STACK_TYPE.append(ST)

#-----Training code------
if TESTING==False:
        if os.path.isfile(CHECKPOINT_NAME):
                os.remove(CHECKPOINT_NAME)
        device = torch.device('cpu')  # use the trainer.py to run on GPU.
		
        data = pd.read_csv(INPUT_FILE_PATH, index_col=0, parse_dates=True)
        data = data.values  # just keep np array here for simplicity.
        norm_constant = args.norm_constant if args.norm_constant!=None else np.max(data) #norm_constant for normalizing the data to lie between 0 and 1
        model_dict = {"norm_constant": norm_constant}
        f = open(DICT_PATH,"wb")
        pickle.dump(model_dict, f)
        f.close()

        print("\nNormalizing constant for the data is: " + str(norm_constant))
        data = data / norm_constant  # small leak to the test set here.
        #print(data)
            
        min_loss = 1000 #Startin minimum loss with a large value
        hiddenLayerUnits = HIDDEN_LAYER_UNITS
        #stackTypes = [[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK], [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK]] #In case you change this please change ThetaDims in the inner most for loop mentioned below.
        stackTypes = STACK_TYPE
        thetaDims = THETA_DIMS
        numBlocksPerStack = BLOCKS_PER_STACK # Number of blocks per stack
        backcastDays = BACKCAST_DAYS # Backcast length of 1-week, 2-weeks, 3-weeks
        batch_size = BATCH_SIZE
        shareWeightsInStack = SHARE_WEIGHTS_IN_STACK
        for days in backcastDays:
                backcast_length = days * 96
                forecast_length = FORECAST_DAYS * 96 #forecast length is set for a forecast of 1 day
                x_train, y_train = [], []
                x_val, y_val = [], []
                #x_test, y_test = [], []

                train_length = ceil((len(data)/96) * (1-VALIDATION_RATIO)) * 96
                val_length = ceil((len(data)/96) * (VALIDATION_RATIO)) * 96
                #test_length = len(data) - train_length - val_length
                print("\nLengths:\nFull: " + str(len(data)) + " Train: " + str(train_length) + " Val: " + str(val_length))
    
                #Divide the dataset here
                #Forming train-set in increments of 1 entry
                for i in range(backcast_length, train_length - forecast_length):
                        x_train.append(data[i - backcast_length:i])
                        y_train.append(data[i:i + forecast_length])
				
    	        #Forming validation-set in increments of 1 day i.e., 96 entries
                for i in range(train_length + backcast_length, len(data) - forecast_length, 96):
                        x_val.append(data[i - backcast_length:i])
                        y_val.append(data[i:i + forecast_length])

                x_train, y_train = np.array(x_train)[...,0], np.array(y_train)[...,0]
                x_val, y_val = np.array(x_val)[...,0], np.array(y_val)[...,0]
				
                #Printing shape of the formed train, test, and validation datasets
                print("\nShape of Resultant Time Series")
                print("Backcast of train & val: " + str(x_train.shape) + ", " + str(x_val.shape))
                print("Forecast of train & val: " + str(y_train.shape) + ", " + str(y_val.shape))
                
                print("\nTraining starting....")
                for a in range(len(stackTypes)):
                    for b in range(len(numBlocksPerStack)):
                        for c in range(len(hiddenLayerUnits)):
                            print("Iteration: " + "Backcast Days: " + str(days) + ", Stack Types: " + str(stackTypes[a]) + ", No. Blocks per Stack: " + str(numBlocksPerStack[b]) + ", Hidden Layer Units: " + str(hiddenLayerUnits[c]))

                            #Forming model by passing model parameters
                            net = NBeatsNet(device = device, stack_types=stackTypes[a], nb_blocks_per_stack=numBlocksPerStack[b], forecast_length=forecast_length, backcast_length=backcast_length, thetas_dims=thetaDims[a], share_weights_in_stack=shareWeightsInStack, hidden_layer_units=hiddenLayerUnits[c])
                            optimiser = optim.Adam(net.parameters())
                            #Save a temporary model to memorize its params and store it in a dictionary. Refer save_temp function
                            save(net,optimiser,0,CHECKPOINT_NAME)
                            
                            # data generator forms by stocking up batch_size mnumber of time series together
                            data = data_generator(x_train, y_train, batch_size)

			    # training
			    # model seems to converge well around ~2500 grad steps and starts to overfit a bit after.
                            test_losses = [] #Used to append test_losses at each validation step
                            for i in range(100):
                                eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_val, y_val)
                                flag = train_200_grad_steps(data, device, net, optimiser, test_losses)
                                    
				##Below code is used for early-stopping the model. flag=0 means the model is to be early-stopped otherwise flag=1
                                if flag==0:
                                    global_step = load(net,optimiser,CHECKPOINT_NAME)
				    #Open this to specify best model by parameters
                                    if MODEL_PREFIX[-3:]==".th": MODEL_PREFIX = MODEL_PREFIX[:-3]
                                    model_dict[MODEL_PREFIX+"[" + str(days) + "]_" + str(stackTypes[a]) + "_" + str(numBlocksPerStack[b]) + "_" + str(hiddenLayerUnits[c]) + "_" + str(test_losses[-1]) + ".th"] = [stackTypes[a], numBlocksPerStack[b], forecast_length, backcast_length, thetaDims[a], shareWeightsInStack, hiddenLayerUnits[c], batch_size]
                                    f = open(DICT_PATH,"wb")
                                    pickle.dump(model_dict, f)
                                    f.close()

                                    save(net ,optimiser, global_step, MODEL_PREFIX+"[" + str(days) + "]_" + str(stackTypes[a]) + "_" + str(numBlocksPerStack[b]) + "_" + str(hiddenLayerUnits[c]) + "_" + str(test_losses[-1]) + ".th")
                                    print("Model saved as: " + MODEL_PREFIX+"[" + str(days) + "]_" + str(stackTypes[a]) + "_" + str(numBlocksPerStack[b]) + "_" + str(hiddenLayerUnits[c]) + "_" + str(test_losses[-1]) + ".th")
                                    print("Stopped Early!")
                                    break
																		
#------Testing code------																	
else:
        if os.path.isfile(CHECKPOINT_NAME):
	        os.remove(CHECKPOINT_NAME)
        device = torch.device('cpu')  # use the trainer.py to run on GPU.

        #Reading model dictionary
        f = open(DICT_PATH,"rb")
        model_dict = pickle.load(f)
        print(model_dict.keys())

        data = pd.read_csv(INPUT_FILE_PATH, index_col=0, parse_dates=True)
        data = data.values  # just keep np array here for simplicity.
        norm_constant = model_dict["norm_constant"] #norm_constant for normalizing the data to lie between 0 and 1
        print("Normalizing constant for the data is: " + str(norm_constant))
        data = data / norm_constant  # small leak to the test set here.
        #print(data)
		
        backcast_length = model_dict[MODEL][3]
        forecast_length = model_dict[MODEL][2]
        x_test, y_test = [], []
		
        test_length = len(data)
        print("Test length: " + str(test_length))
		
        #Forming test-set in increments of 1 day i.e., 96 entries
        for i in range(backcast_length, test_length - forecast_length, 96):
            x_test.append(data[i - backcast_length:i])
            y_test.append(data[i:i + forecast_length])
		
        x_test, y_test = np.array(x_test)[...,0], np.array(y_test)[...,0]
		
        #Printing shape of the formed train, test, and validation datasets
        print("\nShape of Resultant Time Series")
        print("Backcast of test: " + str(x_test.shape))
        print("Forecast of test: " + str(y_test.shape))
		
        hiddenLayerUnits = model_dict[MODEL][6]
        stackTypes = model_dict[MODEL][0] #In case you change this please change ThetaDims in the inner most for loop mentioned below.
        thetaDims = model_dict[MODEL][4]
        numBlocksPerStack = model_dict[MODEL][1] # Number of blocks per stack
        batch_size = model_dict[MODEL][7]
        shareWeightsInStack = model_dict[MODEL][5]
		
        net = NBeatsNet(device = device, stack_types=stackTypes, nb_blocks_per_stack=numBlocksPerStack, forecast_length=forecast_length, backcast_length=backcast_length, thetas_dims=thetaDims, share_weights_in_stack=shareWeightsInStack, hidden_layer_units=hiddenLayerUnits)
        optimiser = optim.Adam(net.parameters())
        
        print("\nLoading model... " + str(MODEL))
        load(net,optimiser,MODEL)
		
        net.eval()
        _, forecast = net(torch.tensor(x_test, dtype = torch.float))
        test_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype = torch.float)).item()
        print("The MSE loss while testing: " +str(test_loss))
        p = forecast.detach().numpy()
		
        if COMPARISON_PLOTS:
            print("Storing prediction files and comparison plots....")
        else:
            print("Storing prediction files....")
				
        for i in range(len(x_test)):
            print("Test series #" + str(i))
            ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
            preds = pd.DataFrame(data = yy, columns = ['y'])
            preds['yhat'] = ff
            preds.to_csv(OUTPUT_DIRECTORY + "pred_" + str(i) + ".csv", index = False)
            
            if COMPARISON_PLOTS:
                plt.figure(1)
                plt.grid()
                plot_scatter(range(0, 96), yy, color='g')
                plot_scatter(range(0, 96), ff, color='r')
                plt.savefig(OUTPUT_DIRECTORY+"Comparison_Plots/complot_" + str(i) + ".png")
                plt.clf()
