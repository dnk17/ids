#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import pickle as pk
from IPython.display import clear_output
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import scale


# # Logistic Regression

# In[ ]:


class LogReg:
    def __init__(self, matrices, predictors, exclude, min_NLang):
        ''' '''
        print('Initiating process...')
        self.matrices = matrices
        self.predictors = predictors
        self.exclude = exclude 
        self.min_NLang = min_NLang
        self.size = matrices['135'].shape
        self.colex_ind, self.zero_set = self.pairs_set()
        clear_output(wait=True)
        print('Initialization done!')
        
        
    def binarize(self):
        """
        Input: set of 279 colexification matrices 
        Output: sum of 279 binarized colexification matrices 
        
        If there is one language that colexifies a pair: 
                entry = 1 
           else:
                entry = 0""" 
        
        individual_binarize = {}
        for i in self.matrices:
            if type(self.matrices[i]) == type(self.matrices['207']):
                ind = np.nonzero(self.matrices[i])
                mat = np.zeros(self.size)
                mat[ind] = 1 
                individual_binarize[i] = mat
        return(sum(individual_binarize.values()))
    
    
    def pairs_set(self):
        """
        Input: 
           - sum of 279 binarized colexification matrices
             dtype: numpy array 
           - indices of pairs that are identical in English
             dtype: panda dataframe 
        
        Output: 
           - indices of colexified pairs (colex_ind )
           - indices of non-colexified pairs (zero_set)
        """
        
        print('Calculating attested set and null set...')
        sumMat = self.binarize()
        subs_set = set([])
        
        for ex in self.exclude: 
            ex_set = set([eval(i) for i in ex.iloc[:,0]])
            subs_set = subs_set | ex_set
        
        upper = set(zip(*np.triu_indices(self.size[0])))
        diagonal = set([(i,i) for i in range(self.size[0])])
        colexified = np.argwhere(sumMat >= self.min_NLang)
        nonzero = set([tuple(i) for i in colexified])
        colex_set = (nonzero&upper) - diagonal - subs_set
        colex_ind = np.array(list(colex_set))
        colex_ind = (colex_ind[:,0],colex_ind[:,1])
        zero_set = upper - colex_set - subs_set - diagonal
        return(colex_ind, zero_set)
    
    def sample_data(self, matrix, samp_multi = False ):
        """
        Sample non-colexified pairs from predictor data 
        
        Input: 
            - matrix: Predictor data (numpy array)
            - samp_multi: True if multivariate regression (boolean) 
            
        Output: 
            - array of sampled data from predictor(s)
        """
        sample_zero = random.sample(list(self.zero_set),len(self.colex_ind[0]))
        sample_zero = np.array(list(sample_zero))
        sample_zero = (sample_zero[:,0],sample_zero[:,1])
            
        if samp_multi == True:
            data = []
            for i in matrix:
                d = np.array(list(i.toarray()[self.colex_ind]) + 
                    list(list(i.toarray()[sample_zero])))
                data.append(scale(d))
            data = np.array(list(zip(*data)))
    
        else:
            data = np.array(list(matrix.toarray()[self.colex_ind]) + 
                    list(list(matrix.toarray()[sample_zero])))
            data = data.reshape(-1,1)
                    
        return(data)
        
    def n_fold(self, data, target): 
        """
        Return mean accuracy of n-fold logistic regression
        
        Input: 
            - data: predictor (numpy array)
            - target: labels (numpy array of 1s and 0s)
            
        Output:
            - mean accuracy score
        """
        acc = [] 
        for fold in range(20):
            X_train, X_test, y_train, y_test =             train_test_split(data, target, test_size=0.1)
            logistic = linear_model.LogisticRegression()
            acc.append(logistic.fit(X_train, y_train).score(X_test, y_test))
        return(np.mean(acc))
    
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
        else:
            outputs = [[] for _ in range(len(self.predictors))]
        
        target = np.array([1]*len(self.colex_ind[0]) + [0]*len(self.colex_ind[0]))
        target = target.reshape(-1,1)

        for rep in range(100):
            clear_output(wait=True)
            print('Calculating Spearman R...')
            print('Sample #: {}'.format(rep))
    
            for i in range(len(self.predictors)):
                data = self.sample_data(self.predictors[i], samp_multi = False)
                acc = self.n_fold(data, target)
                outputs[i].append(acc)
            
            if r_multi == True:
                data = self.sample_data(self.predictors, samp_multi = True)
                mult_acc = self.n_fold(data, target)
                outputs[5].append(mult_acc)
            
        return([round(np.mean(i),3) for i in outputs],
               [round(np.std(i),3) for i in outputs])
    
    def coefficients(self):
        '''
        Calculate the coefficients of each predictor 
        Output:
            - list of the mean coefficients, for each predictor
            - list of std of the coefficients
        '''
        coeff = [] 
        
        target = np.array([1]*len(self.colex_ind[0]) + [0]*len(self.colex_ind[0]))
        target = target.reshape(-1,1)

        for rep in range(100):
            clear_output(wait=True)
            print('Calculating coefficients...')
            print('Sample #: {}'.format(rep))
            data = self.sample_data(self.predictors, samp_multi = True)
        
            logistic = linear_model.LogisticRegression()
            logistic.fit(data, target)
    
            coeff.append(logistic.coef_[0])
        
        coeff = np.array(coeff)
        return([round(np.mean(coeff[:,i]),3) for i in range(5)],
               [round(np.std(coeff[:,i]),3) for i in range(5)])


# # Plot function

# In[37]:


import matplotlib as mpl
mpl.use('TkAgg')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pylab import rcParams

sns.set_style('whitegrid')
sns.set_style("ticks", {"xtick.major.size": 0, "ytick.major.size": 8})

params = {'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'axes.labelsize': 10, 
          'axes.titlesize': 12,
          'font.size': 12, 
          'legend.fontsize': 12, 
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'ytick.direction':'in',
          'figure.figsize': [10,5],
          'font.family': 'serif',}

mpl.pyplot.rcParams.update(params)


# In[54]:


def plot(mean_r, std_r, mean_coeff, std_coeff, title):
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
    mpl.pyplot.ylabel('Predictive Accuracy', size = 12)
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
    mpl.pyplot.ylabel('Coefficient from Logistic Regression', size = 12)
    mpl.pyplot.tight_layout()
    ax3.set_title('B', loc = 'left', fontsize = 25, color = '#1F2060')

    mpl.pyplot.savefig('{}.eps'.format(title), format='eps', dpi=600,bbox_inches='tight')
    mpl.pyplot.show()

