# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Lab 01: Vanilla RNN - demo

# %%
# For Google Colaboratory
import sys, os
if 'google.colab' in sys.modules:
    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # find automatically the path of the folder containing "file_name" :
    file_name = 'vrnn_demo.ipynb'
    import subprocess
    path_to_file = subprocess.check_output('find . -type f -name ' + str(file_name),
                                           shell=True).decode("utf-8")
    path_to_file = path_to_file.replace(file_name, "").replace('\n', "")
    # if previous search failed or too long, comment the previous line and simply write down manually the path below :
    #path_to_file = '/content/gdrive/My Drive/CE7454_2020_codes/codes/labs_lecture10/lab01_vrnn'
    print(path_to_file)
    # change current path to the folder containing "file_name"
    os.chdir(path_to_file)
    get_ipython().system('pwd')

# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time
import utils

# %% [markdown]
# ### With or without GPU?
#
# It is recommended to run this code on GPU:<br>
# * Time for 1 epoch on CPU : 153 sec ( 2.55 min)<br>
# * Time for 1 epoch on GPU : 8.4 sec w/ GeForce GTX 1080 Ti <br>

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %% [markdown]
# ### Download Penn Tree Bank
#
# The tensor train_data consists of 20 columns of 46,479 words.<br>
# The tensor test_data consists of 20 columns of 4,121 words.

# %%
from utils import check_ptb_dataset_exists
data_path = check_ptb_dataset_exists()

train_data = torch.load(data_path + 'ptb/train_data.pt')
test_data = torch.load(data_path + 'ptb/test_data.pt')

print(train_data.size())
print(test_data.size())

# %% [markdown]
# ### Some constants associated with the data set

# %%
bs = 20

vocab_size = 10000

# %% [markdown]
# ### Make a recurrent net class


# %%
class three_layer_recurrent_net(nn.Module):
    def __init__(self, hidden_size):
        super(three_layer_recurrent_net, self).__init__()

        self.layer1 = nn.Embedding(vocab_size, hidden_size)
        self.layer2 = nn.RNN(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_seq, h_init):

        g_seq = self.layer1(word_seq)
        h_seq, h_final = self.layer2(g_seq, h_init)
        score_seq = self.layer3(h_seq)

        return score_seq, h_final


# %% [markdown]
# ### Build the net. Choose the hidden size to be 150. How many parameters in total?

# %%
hidden_size = 150

net = three_layer_recurrent_net(hidden_size)

print(net)

utils.display_num_param(net)

# %% [markdown]
# ### Send the weights of the networks to the GPU

# %%
net = net.to(device)

# %% [markdown]
# ### Set up manually the weights of the embedding module and Linear module

# %%
net.layer1.weight.data.uniform_(-0.1, 0.1)

net.layer3.weight.data.uniform_(-0.1, 0.1)

print('')

# %% [markdown]
# ### Choose the criterion, as well as the following important hyperparameters:
# * initial learning rate = 1
# * sequence length = 35

# %%
criterion = nn.CrossEntropyLoss()

my_lr = 1

seq_length = 35

# %% [markdown]
# ### Function to evaluate the network on the test set


# %%
def eval_on_test_set():

    running_loss = 0
    num_batches = 0

    h = torch.zeros(1, bs, hidden_size)

    h = h.to(device)

    for count in range(0, 4120 - seq_length, seq_length):

        minibatch_data = test_data[count:count + seq_length]
        minibatch_label = test_data[count + 1:count + seq_length + 1]

        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        scores, h = net(minibatch_data, h)

        minibatch_label = minibatch_label.view(bs * seq_length)
        scores = scores.view(bs * seq_length, vocab_size)

        loss = criterion(scores, minibatch_label)

        h = h.detach()

        running_loss += loss.item()
        num_batches += 1

    total_loss = running_loss / num_batches
    print('test: exp(loss) = ', math.exp(total_loss))


# %% [markdown]
# ### Do 10 passes through the training set (100 passes would reach 135 on test set)

# %%
start = time.time()

for epoch in range(10):

    # keep the learning rate to 1 during the first 4 epochs, then divide by 1.1 at every epoch
    if epoch >= 4:
        my_lr = my_lr / 1.1

    # create a new optimizer and give the current learning rate.
    optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    # set the running quantities to zero at the beginning of the epoch
    running_loss = 0
    num_batches = 0

    # set the initial h to be the zero vector
    h = torch.zeros(1, bs, hidden_size)

    # send it to the gpu
    h = h.to(device)

    for count in range(0, 46478 - seq_length, seq_length):

        # Set the gradients to zeros
        optimizer.zero_grad()

        # create a minibatch
        minibatch_data = train_data[count:count + seq_length]
        minibatch_label = train_data[count + 1:count + seq_length + 1]

        # send them to the gpu
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        # Detach to prevent from backpropagating all the way to the beginning
        # Then tell Pytorch to start tracking all operations that will be done on h and c
        h = h.detach()
        h = h.requires_grad_()

        # forward the minibatch through the net
        scores, h = net(minibatch_data, h)

        # reshape the scores and labels to huge batch of size bs*seq_length
        scores = scores.view(bs * seq_length, vocab_size)
        minibatch_label = minibatch_label.view(bs * seq_length)

        # Compute the average of the losses of the data points in this huge batch
        loss = criterion(scores, minibatch_label)

        # backward pass to compute dL/dR, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: R=R-lr(dL/dR), V=V-lr(dL/dV), ...
        utils.normalize_gradient(net)
        optimizer.step()

        # update the running loss
        running_loss += loss.item()
        num_batches += 1

    # compute stats for the full training set
    total_loss = running_loss / num_batches
    elapsed = time.time() - start

    print('')
    print('epoch=', epoch, '\t time=', elapsed, '\t lr=', my_lr, '\t exp(loss)=',
          math.exp(total_loss))
    eval_on_test_set()

# %% [markdown]
# ### Choose one sentence (taken from the test set)

# %%
sentence1 = "some analysts expect oil prices to remain relatively"

sentence2 = "over the next days and weeks they say investors should look for stocks to"

sentence3 = "prices averaging roughly $ N a barrel higher in the third"

sentence4 = "i think my line has been very consistent mrs. hills said at a news"

sentence5 = "this appears particularly true at gm which had strong sales in"

# or make your own sentence.  No capital letter or punctuation allowed. Each word must be in the allowed vocabulary.
sentence6 = "he was very"

# SELECT THE SENTENCE HERE
mysentence = sentence1

# %% [markdown]
# ### Convert the sentence into a vector, then send to GPU

# %%
minibatch_data = utils.sentence2vector(mysentence)

minibatch_data = minibatch_data.to(device)

print(minibatch_data)

# %% [markdown]
# ### Set the initial hidden state to zero, then run the RNN.

# %%
h = torch.zeros(1, 1, hidden_size)
h = h.to(device)

scores, h = net(minibatch_data, h)

# %% [markdown]
# ### Display the network prediction for the next word

# %%
print(mysentence, '... \n')

utils.show_next_word(scores)

# %%
