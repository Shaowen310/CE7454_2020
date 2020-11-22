# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
import time
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from IPython import get_ipython

# %% [markdown]
# # Lab 03 : LeNet5 architecture - exercise

# %%
# For Google Colaboratory
if 'google.colab' in sys.modules:
    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # find automatically the path of the folder containing "file_name" :
    file_name = 'lenet5_exercise.ipynb'
    import subprocess
    path_to_file = subprocess.check_output('find . -type f -name ' + str(file_name),
                                           shell=True).decode("utf-8")
    path_to_file = path_to_file.replace(file_name, "").replace('\n', "")
    # if previous search failed or too long, comment the previous line and simply write down manually the path below :
    #path_to_file = '/content/gdrive/My Drive/CE7454_2020_codes/codes/labs_lecture08/lab03_lenet5'
    print(path_to_file)
    # change current path to the folder containing "file_name"
    os.chdir(path_to_file)
    get_ipython().system('pwd')

# %%

# %% [markdown]
# ### With or without GPU?
#
# It is recommended to run this code on GPU:<br>
# * Time for 1 epoch on CPU : 96 sec (1.62 min)<br>
# * Time for 1 epoch on GPU : 2 sec w/ GeForce GTX 1080 Ti <br>

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %% [markdown]
# ### Download the MNIST dataset

# %%
from utils import check_mnist_dataset_exists
data_path = check_mnist_dataset_exists()

train_data = torch.load(data_path + 'mnist/train_data.pt')
train_label = torch.load(data_path + 'mnist/train_label.pt')
test_data = torch.load(data_path + 'mnist/test_data.pt')
test_label = torch.load(data_path + 'mnist/test_label.pt')

print(train_data.size())
print(test_data.size())

# %% [markdown]
# ### Compute average pixel intensity over all training set and all channels

# %%
mean = train_data.mean()

print(mean)

# %% [markdown]
# ### Compute standard deviation

# %%
std = train_data.std()

print(std)

# %% [markdown]
# ### Make a LeNet5 convnet class.


# %%
class LeNet5_convnet(nn.Module):
    def __init__(self):

        super().__init__()

        # CL1:   28 x 28  -->    50 x 28 x 28
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3, padding=1)

        # MP1: 50 x 28 x 28 -->    50 x 14 x 14
        self.pool1 = nn.MaxPool2d(2, 2)

        # CL2:   50 x 14 x 14  -->    100 x 14 x 14
        self.conv2 = nn.Conv2d(50, 100, 3, padding=1)

        # MP2: 100 x 14 x 14 -->    100 x 7 x 7
        self.pool2 = nn.MaxPool2d(2, 2)

        # LL1:   100 x 7 x 7 = 4900 -->  100
        self.linear1 = nn.Linear(4900, 100)

        # LL2:   100  -->  10
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):

        # CL1:   28 x 28  -->    50 x 28 x 28
        x = self.conv1(x)
        x = F.relu(x)

        # MP1: 50 x 28 x 28 -->    50 x 14 x 14
        x = self.pool1(x)

        # CL2:   50 x 14 x 14  -->    100 x 14 x 14
        x = self.conv2(x)
        x = F.relu(x)

        # MP2: 100 x 14 x 14 -->    100 x 7 x 7
        x = self.pool2(x)

        # LL1:   100 x 7 x 7 = 4900  -->  100
        x = x.view(-1, 4900)
        x = self.linear1(x)
        x = F.relu(x)

        # LL2:   4900  -->  10
        x = self.linear2(x)

        return x


# %% [markdown]
# ### Build the net. How many parameters in total?

# %%
net = LeNet5_convnet()
print(net)
utils.display_num_param(net)

# %% [markdown]
# ### Send the weights of the networks to the GPU (as well as the mean and std)

# %%
net = net.to(device)

mean = mean.to(device)

std = std.to(device)

# %% [markdown]
# ### Choose the criterion, batch size, and initial learning rate. Select the following:
# * batch size =128
# * initial learning rate =0.25

# %%
criterion = nn.CrossEntropyLoss()

my_lr = 0.25

bs = 128

# %% [markdown]
# ### Function to evaluate the network on the test set


# %%
def eval_on_test_set():

    running_error = 0
    num_batches = 0

    for i in range(0, 10000, bs):

        minibatch_data = test_data[i:i + bs].unsqueeze(dim=1)
        minibatch_label = test_label[i:i + bs]

        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        inputs = (minibatch_data - mean) / std

        scores = net(inputs)

        error = utils.get_error(scores, minibatch_label)

        running_error += error.item()

        num_batches += 1

    total_error = running_error / num_batches
    print('error rate on test set =', total_error * 100, 'percent')


# %% [markdown]
# ### Do 30 passes through the training set. Divide the learning rate by 2 every 5 epochs.

# %%
start = time.time()

for epoch in range(1, 30):

    if not epoch % 5:
        my_lr = my_lr / 1.5

    optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    running_loss = 0
    running_error = 0
    num_batches = 0

    shuffled_indices = torch.randperm(60000)

    for count in range(0, 60000, bs):

        # FORWARD AND BACKWARD PASS
        optimizer.zero_grad()

        indices = shuffled_indices[count:count + bs]
        minibatch_data = train_data[indices].unsqueeze(dim=1)
        minibatch_label = train_label[indices]

        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        inputs = (minibatch_data - mean) / std

        inputs.requires_grad_()

        scores = net(inputs)

        loss = criterion(scores, minibatch_label)

        loss.backward()

        optimizer.step()

        # COMPUTE STATS
        running_loss += loss.detach().item()

        error = utils.get_error(scores.detach(), minibatch_label)
        running_error += error.item()

        num_batches += 1

    # AVERAGE STATS THEN DISPLAY
    total_loss = running_loss / num_batches
    total_error = running_error / num_batches
    elapsed = (time.time() - start) / 60

    print('epoch=', epoch, '\t time=', elapsed, 'min', '\t lr=', my_lr, '\t loss=', total_loss,
          '\t error=', total_error * 100, 'percent')
    eval_on_test_set()

# %% [markdown]
# ### Choose image at random from the test set and see how good/bad are the predictions

# %%
# choose a picture at random
idx = randint(0, 10000 - 1)
im = test_data[idx]

# diplay the picture
utils.show(im)

# send to device, rescale, and view as a batch of 1
im = im.to(device)
im = (im - mean) / std
im = im.view(1, 28, 28).unsqueeze(dim=1)

# feed it to the net and display the confidence scores
scores = net(im)
probs = F.softmax(scores, dim=1)
utils.show_prob_mnist(probs.cpu())

# %%
