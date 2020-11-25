# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Lab 02 : Pooling layer - demo

# %%
import torch
import torch.nn as nn

# %% [markdown]
# ### Make a pooling module
# * inputs:  activation maps of size n x n
# * output:  activation maps of size n/p x n/p
# * p: pooling size

# %%
mod = nn.MaxPool2d(2, 2)

# %% [markdown]
# ### Make an input 2 x 6 x 6  (two channels, each one has 6 x 6 pixels )

# %%
bs = 1

x = torch.rand(bs, 2, 6, 6)

print(x)
print(x.size())

# %% [markdown]
# ### Feed it to the pooling layer: the output size should be divided by 2

# %%
y = mod(x)

print(y)
print(y.size())

# %%
