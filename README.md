# Practica Kaggle APC UAB 2021-2022

### Nom: Goretti Pena Lorente
### DATASET: Trip Pricing with Taxi Mobility Analytics
### URL: https://www.kaggle.com/arashnic/taxi-pricing-with-mobility-analytics

## Resum
El Dataset utilitza dades de ...
Tenim 131662 dades amb 14 atributs. Un 39% d'ells és categoric, els altres són numérics. Després d'eliminar Null-values i normalitzar les dades, tenim 111469 dades.

## Objectius del Dataset
Volem aprendre quina és la ...

## Experiments
Durant aquesta pràctica hem realitzat diferents experiments

## Preprocessat
Quines proves hem realitzat que tinguin a veure amb el pre-processat? Com han afectat als resultats?

## Model
| Model | Accuracy | F1-Score | Weighted avg | MSE | Temps |
| -- | -- | -- | -- | -- | -- |
| [Decision Tree](https://github.com/gorettipena/CasKaggle_APC/blob/f70ea986973eaf9fcdc402f5549fdd87c0b86f0f/models/Decision%20Tree.ipynb) | 57% | 0.59 | 0.57 | 4.83 | 13.2 ms |
| [Logistic Regression](https://github.com/gorettipena/CasKaggle_APC/blob/8de608ab68291531e7e24f42955322e8361fa7bd/models/Logistic%20Regression.ipynb) | 43% | 0.60 | 0.26 | 3.67 | 5.98 ms |
| [Random Forest](https://github.com/gorettipena/CasKaggle_APC/blob/5ca79e2cc4ed0dfe4da8cf82326be6d23b0dca21/models/Random%20Forest%20Classifier.ipynb) | 69% | 0.72 |  0.69 |  4.87 | 866 ms |
| [K-Nearest Neighbors](https://github.com/gorettipena/CasKaggle_APC/blob/e7fff479425de040672b99ffe337260564ad3e33/models/KNN.ipynb) | 35% | 0.42 | 0.35 | 4.23 | 37.2 s |
| [Naive-Bayes](https://github.com/gorettipena/CasKaggle_APC/blob/0b4d49cc11baeddd8a712b0ebaebbc6c670bb33a/models/Naive%20Bayes.ipynb) | 52% | 0.64 | 0.43 | 4.27 | 32.6 ms |
| [XGB](https://github.com/gorettipena/CasKaggle_APC/blob/0b4d49cc11baeddd8a712b0ebaebbc6c670bb33a/models/XGB.ipynb) | 70% | 0.79 |  0.70 |  4.86 | 43.3 ms |


## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```

## Conclusions
El millor model que s'ha aconseguit ha estat... 
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....

## Idees per treballar en un futur
Crec que seria interesant indagar més en... 

## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz.
