# Conceptual Relations Predict Colexification across Languages
# (Code for Replication) 

**Abstract**: In natural language, multiple meanings often share a single word form, a phenomenon known as colexification. Some sets of meanings are more frequently colexified across languages than others, but the source of this variation is not well understood. We propose that cross-linguistic variation in colexification frequency reflects the principle of cognitive economy: More commonly colexified meanings across languages are those that require less cognitive effort to relate. To evaluate our proposal, we examine patterns of colexification of varying frequency from about 250 languages. We predict these colexification data based on independent measures of conceptual relatedness drawn from large-scale psychological and linguistic resources. Our results show that meanings that are more frequently colexified across languages tend to be more strongly associated conceptually, suggesting that conceptual associativity provides a key constraint on the development of the lexicon. Our work extends research on polysemy and the evolution of word meanings by grounding cross-linguistic regularities in colexification in basic principles of human cognition.

## Analysis 1
- `logReg_utils.py`: Functions used in Analysis 1. 
- `Analysis1.ipynb`: Predictive accuracies and variabl coefficients of HBC, USF, word2Vec, concreteness, valence, usage frequency and multivariate regression. 

## Analysis 2 
- `linReg_utils.py`: Functions used in Analysis 2. 
- `Analysis2.ipynb`: Spearman ρs and variabl coefficients of HBC, USF, word2Vec, concreteness, valence, usage frequency and multivariate regression. 

## Analysis 3 
- `tiering_utils.py`: Functions used in Analysis 3. 
- `Analysis3.ipynb`: Gradient of colexification frequencies in ordered association sets, using HBC, USF. 

## Folder `data`
### Colexification matrices 
- `colexMats.mat`: Dictionary of all 279 colexification matrices of size 1310 x 1310 from each language included in the analysis.
- `colex_family.mat`, `colex_climate.mat`, `colex_geography.mat`: Dictionary of 1000 bootstrapped colexification matrices of size 1081 x 1081, constructed from the IDS dataset, respectively controlled for family, climate, and geography.
- `MCColex.mat`: Dictionary of 1000 colexification matrices of size 1081 x 1081, constructed from the IDS dataset using Monte Carlo, controlled for family.
- `hbc_colex_family.mat`, `hbc_colex_climate.mat`, `hbc_colex_geography.mat`:(for HBC only) Dictionary of 1000 bootstrapped colexification matrices of size 1191 x 1191, respectively controlled for family, climate, and geography. 
- `hbc_MCColex.mat`: (for HBC only) Dictionary of 1000 colexification matrices of size 1191 x 1191, constructed from the IDS dataset using Monte Carlo, controlled for family.
- `usf_colex_family.mat`, `usf_colex_climate.mat`, `usf_colex_geography.mat`:(for USF only) Dictionary of 1000 bootstrapped colexification matrices of size 1026 x 1026, respectively controlled for family, climate, and geography. 
- `usf_MCColex.mat`: (for USF only) Dictionary of 1000 colexification matrices of size 1026 x 1026, constructed from the IDS dataset using Monte Carlo, controlled for family.

### Controlling factors 
- `languagefamily.csv`: Language family corresponds to each language included in the analysis.
- `macroarea.csv`: Macroarea and climat category correspond to each language included in the analysis. 

### Predictors 
- `hbc.npz`: Association matrix constructed from the HBC dataset. 
- `usf.npz`: Association matrix constructed from the USF dataset. 
- `w2v.npz`: Similarity matrix constructed from the word2Vec dataset. 
- `conc.npz`: Concreteness matrix.
- `val.npz`: Valence matrix.
- `freq.npz`: Usage frequency matrix.
- Folder `hbc`: Folder containing 8 predictor matrices constructed from HBC as specified in Analysis 3.
- Folder `usf`: Folder containing 8 predictor matrices constructed from USF as specified in Analysis 3.

### Excluded indices 
- `exclude.csv`: Indices of senses excluded to obtain the colexification matrices.
- `hbc_exclude.csv`: Indices of senses excluded to obtain the HBC colexification matrices.
- `usf_exclude.csv`: Indices of senses excluded to obtain the USF colexification matrices.

### Excluded pairs 
- `english.csv`: Indices of the pairs of senses that are similar in English.
- `superordinate_pairs.csv`: Indices of superordinate pairs.
- `hbc_english.csv`: (for HBC only) Indices of the pairs of senses that are similar in English 
- `hbc_superordinate.csv`: (for HBC only) Indices of superordinate pairs.
- `usf_english.csv`: (for USF only) Indices of the pairs of senses that are similar in English
- `usf_superordinate.csv`: (for USF only) Indices of superordinate pairs.

### Included pairs
- `nonzeros.csv`: Indices of non-zero colexified pairs across languages. 
- `hbc_nonzeros.csv`: (for HBC only) Indices of non-zero colexified pairs across languages. 
- `usf_nonzeros.csv`: (for USF only) Indices of non-zero colexified pairs across languages.

### Dictionaries 
- `dictionary.csv`: Dictionary of the IDS senses corresponding to each row of the colexification matrix (i.e., `colex_family.mat`, `colex_climate.mat`, `colex_geography.mat`, `MCColex.mat`)
- `hbc_dictionary.csv`: Dictionary of the IDS senses corresponding to each row of the HBC colexification matrix (i.e., `hbc_colex_family.mat`, `hbc_colex_climate.mat`, `hbc_colex_geography.mat`, `hbc_MCColex.mat`)
- `usf_dictionary.csv`: Dictionary of the IDS senses corresponding to each row of the USF colexification matrix (i.e., `usf_colex_family.mat`, `usf_colex_climate.mat`, `usf_colex_geography.mat`, `usf_MCColex.mat`)

