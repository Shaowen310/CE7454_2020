# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Lab 01 : Convolutional layer - demo

# %%
import torch
import torch.nn as nn

# %% [markdown]
# ### Make a convolutional module
# * inputs:  2 channels
# * output:  5 activation maps
# * filters are 3x3
# * padding with one layer of zero to not shrink anything

# %%
mod = nn.Conv2d(2, 5, kernel_size=3, padding=1)

# %% [markdown]
# ### Make an input 2 x 6 x 6  (two channels, each one has 6 x 6 pixels )

# %%
bs = 1

x = torch.rand(bs, 2, 6, 6)

print(x)

print(x.size())

# %% [markdown]
# ### Feed it to the convolutional layer: the output should have 5 channels (each one is 6x6)

# %%
y = mod(x)

print(y)

print(y.size())

# %% [markdown]
# ### Lets look at the 5 filters.
# * Our filters are 2x3x3
# * Each of the filter has 2 channels because the inputs have two channels

# %%
print(mod.weight)

print(mod.weight.size())

# %%
