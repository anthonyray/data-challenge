# 16/04/14
- Fix the energy level for wavelets
- Wavelets on **spectral decomposition**


- New features :
  - Central Frequency
  - RPEB
  - Itakura distance
  - 1-dimensional reduction features (cf paper korea)
TODO :

- Complete Wavelets features
- Try to equilibrate classes

--> Write a good report

- Faire un modèle en couches qui détecte d'abord le wake state

DOING :
- Completing energy / frequency features


DONE :
- Improve score solely working on classifiers :
  - do not train the final classifier on the whole training set to avoid overfitting
  - try different classifiers

- Visualizing features
- Read paper on EEG features
- Use wavelets coefficients :
  - TOP K : Doesn't yield good results
  - Different types : Haar & DB1
  - Make sure to have very clean spectral decomposition (check the paper)
- Clean architecture

- Try different wavelets : Not really relevant. Haar & DB & Coif1 don't show real differences
