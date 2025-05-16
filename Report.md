# Rapport de la démarche pour un détecteur de tweets haineux ou non

## Recontextualisation de l'exercice:
On nous demande d'entrainer une I.A. à détecté si un tweet est haineux et va blesser la personne à qui s'adresse le tweet ou pas.
Il y a donc plusieurs points névralgique à prendre en considération :
- Il faut un temps de réponse de l'ordre du temps réelle < 200ms
- Il faut faire attention à ne pas détecter un tweet non haineux en haineux afin de ne pas rendre l'expérience utilisateur désagréable.
- Il faut quand même réduire considérablement les tweets haineux (2/3 semble un objectif correct)

## Mise en place des métriques et des choix:
Tout d'abord, on prend le parti pris de se concentrer sur le choix de la métrique à maximiser afin de respecter le 2ème points du paragraphe ci-dessus.
On va choisir 2 métriques possibles : La précision ou le "f1 score" sur les tweets non haineux.
Cependant, il nous faut tout de même détecter 2/3 des tweets haineux au minimum. Il est donc intéressant que les positfs soient les haineux et les négatifs soient les passifs. 
Pour le choix des modèles, il nous faut faire attention à choisir un modèle de type classifier qui va catégoriser les tweets.
Ici on se concentrera sur 3 modèles : 
- "vinai/bertweet-base" de Hugging Face
- "distilbert-base-uncased" de Hugging Face
- RandomForestClassifier qui se trouve dans la librairie sklearn.ensemble

## Dataset choisis et répartition des données:
On choisit le dataset de tweets : cyberbullying_tweets.csv. Il découle de cela un problème majeur :
La répartition des tweets entre haineux ou non. En effet le dataset est sans-doute équilibrer entre non haineux, raciste, ... mais si on considère toute forme de haine comme un seul label alors on a 85% de tweets haineux contre 15% de tweets non haineux.  
La solution à ce problème est de faire des sous-ensemble qui respecte cette répartition.

## Résultats finaux et choix de l'I.A.:
On a alors les résultats suivant:

- "vinai/bertweet-base" détecte 38% des tweets non haineux correctement et 94% des tweets haineux correctement.
- "distilbert-base-uncased" détecte 60% des tweets non haineux correctement et 95% des tweets haineux correctement.
- RandomForestClassifier détecte 94% des tweets non haineux correctement et 74% des tweets haineux correctement.

Le meilleur modèle pour notre cas est donc le RandomForestClassifier.