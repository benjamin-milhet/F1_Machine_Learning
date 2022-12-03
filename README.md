# F1_Machine_Learning

## Planification
![alt text](https://github.com/benjamin-milhet/F1_Machine_Learning/blob/main/Capture.PNG?raw=true)
https://trello.com/b/Ydjs2wPd/projet-4a-formula-1-dataset

## Préparation

### What is the difference between supervised and unsupervised learning ?

https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning

Un apprentissage supervisé possède des données en entrée et en sortie. Ce type d’apprentissage possède un training set, un ensemble de données qui permet d’entraîner notre algorithme avec des données en entre à tester et les différentes solutions associées. Cela lui permet d’avoir une base pour ensuite étudier de nouveaux individus et d’avoir une idée de quelle type de solution l’algorithme doit chercher.

Pour l’apprentissage non-supervisé, il n’y a pas de donnée en sortie mais juste en entrée. C’est à l’algorithme de déduire les points importants et de proposer ses solutions sans avoir été entraîné auparavant.


### What is the difference between classification and regression? Is it supervised or unsupervised or unsupervised learning?

La classification et la régression sont 2 types d’algorithmes utilisant un apprentissage supervisé. On utilise la classification lorsque les solutions souhaitées sont des catégories comme des pommes ou des oranges. Alors que la régression est utilisée pour des valeurs numériques comme pour prédire le chiffre d’affaire d’une entreprise. La régression essaye de comprendre les relations entre les différentes variables. 


### What is clustering? What is the difference with classification?

https://developers.google.com/machine-learning/clustering/clustering-algorithms?hl=en

La principale différence entre la classification et le clustering est que la classification utilise un apprentissage supervisé alors que le clustering suit un apprentissage non-supervisé. Le clustering se base sur les similitudes des paires en entrée, et sur son expérience au fur et à mesure de de tester différentes entrée. Le temps d’exécution peut être très élevé si le nombre d’exemple en entrée est très élevé (plusieurs millions)


### Sur kaggle ou driven-data, choisissez 5 exemples de compétitions et dites quel est le type de problème (classification, régression, clustering ou autre)

https://www.kaggle.com/competitions

#### Compétition 1

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

On remarque que le jeu de donnée possède un jeu de données d’entraînement et un jeu de donnée à tester. On est donc sur un apprentissage supervisé. Il faut trouver une valeur numérique pour prédire le prix de vente. Je pense qu’il faut utiliser une régression.


#### Compétition 2

https://www.kaggle.com/competitions/spaceship-titanic/overview

On remarque que le jeu de donnée possède un jeu de données d’entraînement et un jeu de donnée à tester. On est donc sur un apprentissage supervisé. Il faut prédire si le passager va voyager dans une autre dimension ou non. Je pense qu’il faut utiliser une classification parce que c’est un choix binaire.


#### Compétition 3

https://www.kaggle.com/competitions/titanic/overview

On remarque que le jeu de donnée possède un jeu de données d’entraînement et un jeu de donnée à tester. On est donc sur un apprentissage supervisé. Il faut prédire si le passager du Titanic va mourir ou non. Je pense qu’il faut utiliser une classification parce que c’est un choix binaire.


#### Compétition 4

https://www.kaggle.com/competitions/nlp-getting-started

On remarque que le jeu de donnée possède un jeu de données d’entraînement et un jeu de donnée à tester. On est donc sur un apprentissage supervisé. Il faut prédire si l’information dans le tweet est vrai ou non. Je pense qu’il faut utiliser une classification parce que c’est un choix binaire.


#### Compétition 5

https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/

On remarque que le jeu de donnée possède un jeu de données d’entraînement et un jeu de donnée à tester. On est donc sur un apprentissage supervisé. Il faut prédire si n individu va se faire vacciner pour h1n1 ou/et pour la grippe saisonnière ou non. Je pense qu’il faut utiliser une classification parce que c’est un choix binaire, il faut classer les personnes en fonction de quelle vaccin ils vont choisir ou non.



### Quel type de problème de machine learning vous semble le plus répandu ?

Je remarque déjà que le type d’apprentissage le plus utilisé est l’apprentissage supervisé. On en déduit donc 2 types de problèmes qui sont le plus répandu et qui sont la classification et la régression. D’après les exemples précédents, j’ai l’impression que le problème de machine learning le plus répandu est la classification.


## Choix du sujet

https://www.kaggle.com/datasets/thedevastator/formula-one-racing-a-comprehensive-data-analysis

Analyser et prédire le nombre de point d´un pilote à la fin d´une saison -> Régression linéaire multiple.

Pour commencer, nous essayerons déja de déterminer le nombre de point que peut gagner un pilote pour un grand prix choisit.

Ce dataset contient des informations sur l'ensemble des courses de Formule 1 depuis l'année 1950. Il y a plusieurs fichiers des datas sur les pilotes, les tracées des courses, sur les écuries, et l'emble des résultats pour les qualifications, course sprint et (vrai) course. 
Le fichier le plus important est result.csv car il contient les résultats de chaque course avec le score de chaque pilote et le lieux. 
L'objectif est de trouvé un modèle qui permet de prédire le nombre de point d´un pilote à la fin d'une saison. Pour cela, nous devons trouver les variables qui ont le plus d'impact sur le nombre de point d´un pilote.
Il s'agit d'un apprentissage supervisé car nous avons un jeu de données d'entraînement et un jeu de données à tester. Nous devons prédire un nombre de point d'un pilote à la fin d´une saison. Je pense qu'il faut utiliser une régression linéaire parce que nous devons déterminer une valeur numérique.

#### Partie 1 : Algorithmes de régression linéaire simple
J'ai commencé par analyser mes données et sélectionner celle qui me semble importante pour commencer.
En premier lieux j'ai commencé par une régression linéaire simple pour prendre en main les différentes bibliothèques.
J'ai pris comme paramètre le nombre de points d'un pilote par rapport à ses différentes courses.
Les résultats ne sont pas très concluant car le modèle n'est pas très précis. Le nombre de points gagné par un pilote sur une course ne dépend pas uniquement du tracé d'un circuit.

#### Partie 2 : Algorithmes de régression linéaire multiple
Un problème de régression linéaire multiple serait déja plus intéressant car nous avons plusieurs variables qui ont un impact sur le nombre de point d'un pilote. 
Les différentes variables qui me semble importante sont le circuit, l'écurie a laquelle appartient le pilote et durant quelle année, sa position sur la grille de départ et ses différents temps de courses et leurs évolutions.

#### Partie 3 : Chercher d'autre algorithme de regression

