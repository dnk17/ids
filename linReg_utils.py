#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from scipy.io import loadmat
from scipy.stats import spearmanr
import pickle as pk
from IPython.display import clear_output
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import scale


# # Linear Regression

# In[18]:


class LinReg:
    def __init__(self, matrices, raw_predictors, exclude, non_zero):
        ''' '''
        print('Initiating process...')
        self.matrices = matrices
        self.raw_predictors = raw_predictors
        self.exclude = exclude 
        self.size = matrices['135'].shape
        self.non_zero = non_zero
        self.final_ind = self.index_list()
        self.predictors = self.predictor_processing()
        clear_output(wait=True)
        print('Initialization done!')
    

    def index_list(self):
        """
        Input: 
           - indices of pairs that need to be excluded
             dtype: panda dataframe 
        
        Output: 
           - indices of colexified pairs (colex_ind )
           - indices of non-colexified pairs (zero_set)
        """
        
        print('Calculating indices of pairs...')
        subs_set = set([])
        for ex in self.exclude: 
            ex_set = set([eval(i) for i in ex.iloc[:,0]])
            subs_set = subs_set | ex_set
        
        upper = set(zip(*np.triu_indices(self.size[0])))
        diagonal = set([(i,i) for i in range(self.size[0])])
        colexified = set([eval(i) for i in self.non_zero.iloc[:,0]])
        
        final_ind = (upper & colexified) - diagonal - subs_set
        final_ind = np.array(list(final_ind))
        final_ind = (final_ind[:,0],final_ind[:,1])
        return final_ind
    
    
    def predictor_processing(self):
        output = []
        
        for pred in self.raw_predictors:
            arr = pred.toarray()[self.final_ind]
            output.append(scale(arr))
            
        return(output)
    
    
    def spearman_r(self, r_multi = True):
        '''
        Input: 
            - predictor matrices of identical size (numpy array)
            - r_multi: True if multivariate regression (boolean)
        Output: 
            - list of mean accuracy scores for each predictor
            - list of standard deviation of accuracy scores
        '''
        if r_multi == True:
            outputs = [[] for _ in range(len(self.predictors)+1)]
            coeff = [] 
        else:
            outputs = [[] for _ in range(len(self.predictors))]
    
        for rep in range(1000):
            clear_output(wait=True)
            print('Sample #: {}'.format(rep))
            ids = scale(self.matrices[str(rep)].toarray()[self.final_ind])
            
            for i in range(len(self.predictors)):
                rho = spearmanr(self.predictors[i], ids)[0]
                outputs[i].append(rho)
            
            if r_multi == True:
                data = np.array(list(zip(*self.predictors)))
                regr = linear_model.LinearRegression()
                regr.fit(data, ids)
                ids_pred = regr.predict(data)
                rho = spearmanr(ids_pred, ids)[0]
                outputs[5].append(rho)
                coeff.append(regr.coef_)
                
        mean_r = [np.mean(i) for i in outputs]
        std_r = [np.std(i) for i in outputs]
        
        if r_multi == False:
            return(mean_r, std_r)
        
        else:
            coeff = np.array(coeff)
            mean_coeff = [np.mean(coeff[:,i]) for i in range(5)]
            std_coeff = [np.std(coeff[:,i]) for i in range(5)]
            return(mean_r,std_r, mean_coeff, std_coeff)


# # Plot function

# In[37]:


import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
import pandas as pd

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
          'font.family': 'serif',}

mpl.pyplot.rcParams.update(params)


# In[79]:


def  draw_subplots(figure, data, err, i):
    figure.grid(True, which='both', axis = 'y',zorder = 0)

    figure.bar(list(range(6)) + [6.5],
       data,
       align = 'center', width = 0.7, edgecolor = 'none',
       color = ('#ff7538','#c1828d','#19647e', '#d14336', '#048a81', '#ffae03','#b6c649'), zorder =3,
       yerr=err, error_kw=dict(ecolor='dimgray', lw=1, zorder=5))

    figure.set_xlim(-1,7)
    figure.set_ylim(-0.20, 0.3)
    figure.yaxis.set_ticks_position('left')
    figure.xaxis.set_ticks_position('bottom')
    figure.set_yticks(np.arange(-0.1,0.3,0.05))
    if i == 0: 
        figure.set_ylabel('Spearman $\\rho$')


# In[80]:


def  draw_subplots_5(figure, data, err, i):
    figure.grid(True, which='both', axis = 'y',zorder = 0)

    figure.bar(range(5),
       data,
       align = 'center', width = 0.7, edgecolor = 'none',
       color = ('#ff7538','#19647e',  '#d14336','#048a81', '#ffae03'), zorder =3,
       yerr=err, error_kw=dict(ecolor='dimgray', lw=1, zorder=5))

    figure.set_xlim(-1,5)
    figure.set_ylim(-0.05, 0.25)
    figure.yaxis.set_ticks_position('left')
    figure.xaxis.set_ticks_position('bottom')
    figure.set_yticks(np.arange(-0.1,0.23,0.05))
    if i == 0: 
        figure.set_ylabel('Coefficient from Linear Regression')


# In[87]:


def draw_family(means, errors, means_coeff, errors_coeff, plot_title):
    mpl.pyplot.figure(figsize=(15,10))
    fig, axes = mpl.pyplot.subplots(2, 3)
    mpl.pyplot.setp(axes[0], xticks=list(range(6)) + [6.5], 
         xticklabels=('Assoc.\n(HBC)', 'Assoc.\n(USF)',
                                  'Sim.\n(w2v)','Met.\n(Conc.)',
                                  'Met.\n(Val.)', 'Freq.','Combination\nModel'))

    mpl.pyplot.setp(axes[1], xticks=list(range(5)), 
         xticklabels=('Assoc. \n(HBC)','Sim. \n(w2v)','Met. \n(Conc.)',
                                  'Met. \n(Val.)', 'Freq.'))


    for i in range(3): 
        draw_subplots(axes[0][i],means[i],errors[i],i) 

    for i in range(3): 
        draw_subplots_5(axes[1][i],means_coeff[i],errors_coeff[i],i) 
    
    axes[0][0].set_title('Family-controlled', size = 15, ha='center')
    axes[0][1].set_title('Climate-controlled', size = 15, ha='center')
    axes[0][2].set_title('Geography-controlled', size = 15, ha='center')

    axes[0][0].text(-1, 0.31,'A', size = 25, ha='left', color = '#1F2060')
    axes[0][1].text(-1, 0.31,'B', size = 25, ha='left', color = '#1F2060')
    axes[0][2].text(-1, 0.31,'C', size = 25, ha='left', color = '#1F2060')

   
    axes[1][0].set_title('D', loc = 'left', fontsize = 25,color = '#1F2060')
    axes[1][1].set_title('E', loc = 'left', fontsize = 25,color = '#1F2060')
    axes[1][2].set_title('F', loc = 'left', fontsize = 25,color = '#1F2060')

    mpl.pyplot.tight_layout()
    
    mpl.pyplot.savefig('analysis2/{}.eps'.format(plot_title), format='eps', dpi=600)
    mpl.pyplot.show()


# In[88]:


def plot(mean_r, std_r, mean_coeff, std_coeff, title):
    mpl.pyplot.figure(figsize=(10,5))
    ticks = list(range(len(mean_r)-1)) + [len(mean_r)-0.5]
    ax1 = mpl.pyplot.subplot(121)

    ax1.bar(ticks,mean_r,
           align = 'center', edgecolor = 'none',
           color = ('#ff7538','#c1828d','#19647e',  
                    '#d14336','#048a81', '#ffae03','#b6c649'),
           yerr=std_r, error_kw=dict(ecolor='dimgray', lw=1))

    mpl.pyplot.xticks(ticks, ('Assoc.\n(HBC)', 'Assoc.\n(USF)',
                                  'Sim.\n(w2v)','Met.\n(Conc.)',
                                  'Met.\n(Val.)', 'Freq.','Combination\nModel'))

    
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    mpl.pyplot.ylabel('Spearman $\\rho$', size = 12)
    ax1.set_title('A', loc = 'left', fontsize = 25,color = '#1F2060')

    ax3 = mpl.pyplot.subplot(122)
    ax3.bar(np.linspace(0, 2.5, 5),
            mean_coeff,
            align = 'center', width = 0.5, edgecolor = 'none',
            color = ('#ff7538','#19647e','#d14336','#048a81','#ffae03'),
            yerr=std_coeff, error_kw=dict(ecolor='grey', lw=1))
    mpl.pyplot.xticks(np.linspace(0, 2.5, 5), 
                      ('Assoc.\n(HBC)', 'Sim.\n(w2v)','Met.\n(Conc.)',
                   'Met.\n(Val.)', 'Freq.'))
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    mpl.pyplot.ylabel('Coefficient from Linear Regression', size = 12)
    mpl.pyplot.tight_layout()
    ax3.set_title('B', loc = 'left', fontsize = 25, color = '#1F2060')

    mpl.pyplot.savefig('analysis2/{}.eps'.format(title), format='eps', dpi=600,bbox_inches='tight')
    mpl.pyplot.show()

