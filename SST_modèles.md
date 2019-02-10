# Modèles de sentiment analysis sur des phrases structurées

## Introduction / Description
- Ce challenge Kaggle (https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only) correspond en réalité à un jeu de données apparemment assez connu en NLP / Sentiment Analysis, le "Stanford Sentiment Treebank" (https://nlp.stanford.edu/sentiment/index.html).
- Ce jeu de données a été construit à l'aide de critiques de films issues de "Rotten Tomatoes", qui ont été passées dans un outil d'analyse syntaxique du langage naturel, le "Stanford Parser" (https://nlp.stanford.edu/software/lex-parser.shtml).
- Le Stanford Parser transforme une phrase en langage naturel en arbre contenant les différentes propositions dans la phrase (sujet / verbe / complément puis déterminant / nom etc.). Ce pré-traitement a l'air assez intéressant car il permet de donner en entrée aux modèles de ML un langage déjà structuré, plutôt que de chercher à faire en sorte que le modèle apprenne la structure lui-même, ce qui a l'air assez délicat.

## État de l'art / comparaison des modèles existants

Visiblement ce jeu de données est utilisé régulièrement en tant que test des dernières techniques de sentiment analysis. Il y a d'autres jeux de données utilisés pour cela, par exemple, il existe un autre jeu de données réalisé à partir de critiques de films (IMDB) ou même à partir de critiques Yelp.

On peut trouver ici (https://github.com/LINE-PESC/nlp-benchmarks/blob/master/stanford-sentiment-treebank.md) un dépôt github montrant les taux d'erreurs de différentes techniques.

- Les modèles marchent mieux lorsqu'il doivent classer une critique en positive ou négative plutôt que lorsqu'ils doivent donner une échelle allant de 1 à 5 par exemple.
- Un modèle non supervisé, entraîné à partir de critiques Amazon, a obtenu de très bon résultats sur le Stanford Sentiment Treebank : https://blog.openai.com/unsupervised-sentiment-neuron/.  Ce modèle est intéressant car il fait ses prédiction à l'aide d'un seul neurone, mais ce n'est pas un modèle très pertinent pour notre projet car il a été construit avec une puissance de calcul conséquente (même si je n'ai pas eu le temps de tout lire). Le modèle a seulement été entraîné pour prédire un caractère suivant dans le texte d'une critique amazon !
- Plus intéressant pour notre projet : les modèles qui ont l'air de s'en sortir le mieux "prennnent en compte" l'ordre des mots. C'est le cas par exemple des réseaux de neurones récurrents, et plus particulièrement des LSTM (long short-term memory). 
- Le "meilleur" modèle à ce jour sur le Stanford Sentiment Treebank est ici : https://blog.openai.com/block-sparse-gpu-kernels/.
- À voir aussi, ce dépôt github (https://github.com/sebastianruder/NLP-progress) qui renseigne à propos de l'état de l'art en NLP, et plus particulièrement ici : https://github.com/sebastianruder/NLP-progress/blob/master/english/sentiment_analysis.md 
