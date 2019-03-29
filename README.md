# Conceptual Relations Predict Colexification across Languages
# (Code for Replication) 

**Abstract**: In natural language, multiple meanings often share a single word form, a phenomenon known as colexification. Some sets of meanings are more frequently colexified across languages than others, but the source of this variation is not well understood. We propose that cross-linguistic variation in colexification frequency reflects the principle of cognitive economy: More commonly colexified meanings across languages are those that require less cognitive effort to relate. To evaluate our proposal, we examine patterns of colexification of varying frequency from about 250 languages. We predict these colexification data based on independent measures of conceptual relatedness drawn from large-scale psychological and linguistic resources. Our results show that meanings that are more frequently colexified across languages tend to be more strongly associated conceptually, suggesting that conceptual associativity provides a key constraint on the development of the lexicon. Our work extends research on polysemy and the evolution of word meanings by grounding cross-linguistic regularities in colexification in basic principles of human cognition.

## Folder `data`
- `colex_family.mat`, `colex_climate.mat`, `colex_geography.mat`: Dictionary of 1000 bootstrapped colexification matrices, constructed from the IDS dataset, respectively controlled for family, climate, and geography. 
- `hbc_colex_family.mat`, `hbc_colex_climate.mat`, `hbc_colex_geography.mat`:(for HBC only) Dictionary of 1000 bootstrapped colexification matrices, respectively controlled for family, climate, and geography. 
- `usf_colex_family.mat`, `usf_colex_climate.mat`, `usf_colex_geography.mat`:(for USF only) Dictionary of 1000 bootstrapped colexification matrices, respectively controlled for family, climate, and geography. 
- `hbc.npz`: Association matrix constructed from the HBC dataset. 
- `usf.npz`: Association matrix constructed from the USF dataset. 
- `w2v.npz`: Similarity matrix constructed from the word2Vec dataset. 
- `conc.npz`: Concreteness matrix
- `val.npz`: Valence matrix 
- `freq.npz`: Usage frequency matrix 
- `eng_ex.csv`: Indices of the pairs of senses that are similar in English
- `hbc_eng_ex.csv`: (for HBC only) Indices of the pairs of senses that are similar in English 
- `usf_eng_ex.csv`: (for USF only) Indices of the pairs of senses that are similar in English 
- Folder `hbc`: Folder containing 8 predictor matrices constructed from HBC as specified in Analysis 3 
- Folder `usf`: Folder containing 8 predictor matrices constructed from USF as specified in Analysis 3 

## Analysis 1
- `logReg_utils.py`: Functions used in Analysis 1. 
- `Analysis1.ipynb`: Predictive accuracies and variabl coefficients of HBC, USF, word2Vec, concreteness, valence, usage frequency and multivariate regression. 

## Analysis 2 
- `linReg_utils.py`: Functions used in Analysis 2. 
- `Analysis2.ipynb`: Spearman ρs and variabl coefficients of HBC, USF, word2Vec, concreteness, valence, usage frequency and multivariate regression. 

## Analysis 3 
- `tiering_utils.py`: Functions used in Analysis 3. 
- `Analysis3.ipynb`: Gradient of colexification frequencies in ordered association sets, using HBC, USF. 

