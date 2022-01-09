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
| Model | Hiperparametres | Accuracy | F1-Score | Weighted avg | Temps |
| -- | -- | -- | -- | -- | -- |
| [Decision Tree](https://github.com/gorettipena/CasKaggle_APC/blob/fe79da990ffa64f5a9ae8da5f15bb615ec75b600/Decision%20Tree.ipynb) | 1000 Trees, XX | 57% | 0.59 | 0.57 | 13.2 ms |
| [Logistic Regression](https://github.com/gorettipena/CasKaggle_APC/blob/31479c50999e94d14b4db97f90aa0a06042df440/Logistic%20Regression.ipynb) | 1000 Trees, XX | 43% | 0.60 | 0.26 | 5.98 ms |
| [Random Forest](https://github.com/gorettipena/CasKaggle_APC/blob/31479c50999e94d14b4db97f90aa0a06042df440/Random%20Forest%20Classifier.ipynb) | 100 Trees, XX | 69% | 0.72 |  0.69 | 866 ms |
| [K-Nearest Neighbors](https://github.com/gorettipena/CasKaggle_APC/blob/31479c50999e94d14b4db97f90aa0a06042df440/KNN.ipynb) | 1000 Trees, XX | 35% | 0.42 | 0.35 | 37.2 s |
| [Naive-Bayes](https://github.com/gorettipena/CasKaggle_APC/blob/31479c50999e94d14b4db97f90aa0a06042df440/Naive%20Bayes.ipynb) | 1000 Trees, XX | 52% | 0.64 |  0.43 | 32.6 ms |
| SVM | kernel: lineal C:10 | 58% | 200ms |
| -- | -- | -- | -- |
| [model de XXX](link al kaggle) | XXX | 58% | ?ms |
| [model de XXX](link al kaggle) | XXX | 62% | ?ms |

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
