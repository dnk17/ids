#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
import matplotlib as mpl
mpl.use('TkAgg')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
from IPython.display import clear_output
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.io import loadmat

params = {'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'text.usetex': True,
          'axes.labelsize': 15, 
          'axes.titlesize': 12,
          'font.size': 12, 
          'legend.fontsize': 12, 
          'xtick.labelsize': 10,
          'ytick.labelsize': 15,
          'ytick.direction':'in',
          'figure.figsize': [5,5],
          'font.family': 'serif',}

mpl.pyplot.rcParams.update(params)


# In[2]:


def ord_indices(exclude, non_zero, data):
    predictors = []
    for i in range(1,9):
        predictors.append(load_npz('data/{}/{}_pred{}.npz'.format(data, data,i)))

    size = predictors[0].shape[0]
    nonzero_colex = set([eval(i) for i in non_zero.iloc[:,0]])
    upper = set(zip(*np.triu_indices(size)))
    diagonal = set([(i,i) for i in range(size)])
    subs_set = set([])
    for i in exclude:
        mySet = set([eval(i) for i in exclude[0].iloc[:,0]])
        subs_set | mySet 
    ord1_indices =  (set(zip(*np.nonzero(predictors[0])))                      & nonzero_colex  & upper) - diagonal - subs_set
    ord2_indices = (set(zip(*np.nonzero(predictors[1] + predictors[2] + predictors[3])))                    & nonzero_colex  & upper) - ord1_indices                     - diagonal - subs_set
    ord3_indices = (set(zip(*np.nonzero(predictors[4] + predictors[5]                                         + predictors[6]+predictors[7])))                    & nonzero_colex &upper)                    - ord1_indices - ord2_indices - diagonal  - subs_set
    
    ord1_indices = np.array(list(ord1_indices))
    ord1_indices = (ord1_indices[:,0],ord1_indices[:,1])
    
    ord2_indices = np.array(list(ord2_indices))
    ord2_indices = (ord2_indices[:,0],ord2_indices[:,1])
    
    ord3_indices = np.array(list(ord3_indices))
    ord3_indices = (ord3_indices[:,0],ord3_indices[:,1])
    
    return(list((ord1_indices, ord2_indices,ord3_indices)))


# In[3]:


def colex_ord(matrices, ord1_indices, ord2_indices,ord3_indices):
    ord1_l, ord2_l, ord3_l  = [], [], []
    for i in range(1000):
        os.system('clear')
        clear_output(wait=True)
        print('Trial #: ', i)
    
        mat = matrices[str(i)]
        ord1, ord2, ord3 = (mat.toarray()[ord1_indices],
                    mat.toarray()[ord2_indices],
                    mat.toarray()[ord3_indices])
        ord1_l.append(np.mean(ord1))
        ord2_l.append(np.mean(ord2))
        ord3_l.append(np.mean(ord3))
        
    means = [np.mean(ord1_l), np.mean(ord2_l), np.mean(ord3_l)]
    errors = [np.std(ord1_l), np.std(ord2_l), np.std(ord3_l)]
    
    return(means, errors)


# In[4]:


def plot_mc_analysis3(means, errors, plot_title, data):
    mpl.pyplot.grid(True, which='both', axis = 'y',zorder = 0)
    if data == 'hbc':
        colors = ['#ff7538','#ff9e75', '#fcd2bf']
    else:
        colors = ['#c1828d','#d6a4ad', '#ffccd8']
        
    mpl.pyplot.bar(list(range(3)),
               means,
               align = 'center', width = 0.7, edgecolor = 'none',
               color = colors, zorder =2,
               yerr= errors, error_kw=dict(ecolor='dimgray', lw=1, zorder=3))

    mpl.pyplot.xlim(-1,3)
    mpl.pyplot.ylim(0, 0.035)
    mpl.pyplot.yticks(np.arange(0,0.035,0.01))
    mpl.pyplot.ylabel('Colexification Frequency')
    mpl.pyplot.xticks(list(range(3)), ('1st-order \n association',
                      '2nd-order \n association',
                      '3rd-order \n association'))
    mpl.pyplot.savefig('analysis3/{}.eps'.format(plot_title), format='eps', dpi=600)
    mpl.pyplot.show()


# In[5]:


def  draw_subplots(figure, mean, err, i, data):
    figure.grid(True, which='both', axis = 'y',zorder = 0)
    if data == 'hbc':
        colors = ['#ff7538','#ff9e75', '#fcd2bf']
    else:
        colors = ['#c1828d','#d6a4ad', '#ffccd8']
        
    figure.bar(list(range(3)),
               mean,
               align = 'center', width = 0.7, edgecolor = 'none',
               color = colors, zorder =2,
               yerr=err, error_kw=dict(ecolor='dimgray', lw=1, zorder=3))

    figure.set_xlim(-1,3)
    figure.set_ylim(0, 0.045)
    figure.yaxis.set_ticks_position('left')
    figure.xaxis.set_ticks_position('bottom')
    figure.set_yticks(np.arange(0,0.035,0.01))
    if i == 0: 
        figure.set_ylabel('Colexification Frequency')


# In[6]:


def plot_sb_analysis3(means, errors, plot_title, data):
    fig, axes = mpl.pyplot.subplots(1, 3)
    mpl.pyplot.setp(axes, xticks=list(range(3)), 
         xticklabels=('1st-order \n association',
                      '2nd-order \n association',
                      '3rd-order \n association'))

    for i in range(3): 
        draw_subplots(axes[i],means[i],errors[i],i, data) 
    
    axes[0].set_title('Family-controlled', size = 15)
    axes[1].set_title('Climate-controlled', size = 15)
    axes[2].set_title('Geography-controlled', size = 15)

    axes[0].text(-1, 0.046,'A', size = 25, ha='left', color = '#1F2060')
    axes[1].text(-1, 0.046,'B', size = 25, ha='left', color = '#1F2060')
    axes[2].text(-1, 0.046,'C', size = 25, ha='left', color = '#1F2060')

    mpl.pyplot.tight_layout()

    mpl.pyplot.savefig('analysis3/{}.eps'.format(plot_title), format='eps', dpi=600)
    mpl.pyplot.show()


# In[7]:


def sb_analysis3(factors, exclude, non_zero, data, plot_title):
    means = []
    errors = []
    for i in factors: 
        indices = ord_indices(exclude, non_zero, data)
        mean, err = colex_ord(i, *indices)
        means.append(mean)
        errors.append(err)
    mpl.pyplot.rcParams['figure.figsize'] = 15, 5
    plot_sb_analysis3(means, errors, plot_title, data)


# In[8]:


def mc_analysis3(matrices, exclude, non_zero, data, plot_title):
    indices = ord_indices(exclude, non_zero, data)
    means, errors = colex_ord(matrices, *indices)
    mpl.pyplot.rcParams['figure.figsize'] = 5, 5
    plot_mc_analysis3(means, errors, plot_title, data)

