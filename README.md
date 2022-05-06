# nt-challenge

## Introduction

### Rappel de l'énoncé 

    Créer un bot gérant 2 intentions de mise en relation

    Intentions :
        Mise en relation avec un service
            humain : bonjour, je voudrais le service _XY_
            bot : très bien, je vous passe le service _XY_

        Mise en relation avec un docteur
            humain : bonjour, je voudrais parler au docteur _XY_
            bot : très bien, je vous passe le docteur _XY_
    
    NB: ces phrases sont des exemples, le bot doit être capable de comprendre de multiples formes de ces 2 intentions

    Restitution :

        Une démo fonctionnelle et utilisable (pas de vidéo de bon fonctionnement)
        Le code source de la solution
        Tout élément
            permettant de valider la qualité de compréhension du bot
            pertinent de votre point de vue

    Elément mis à disposition :

        un jeu de tests contenant une liste de services et une liste de médecin permettant la mise en oeuvre de ce bot

### Distinction d'intention

Après avoir regardé ce qui se fait avec rasa, mais aussi plus généralement dans le milieu médical, j'ai décidé de me concentrer sur la différenciation des deux intentions.

Pour ce faire :
- J'ai essayé différentes approches pour générer des données d'entrainement pertinentes.
- J'ai essayé différentes pipelines. À ce titre, j'ai essayé de comparer l'utilisation d'un ResponseSelector avec un modèle plus classique basé sur la détection d'entité.

Je n'ai pas du tout travaillé sur la partie "discussion" du chatbot. Les différents modèles ne font donc pas de liens entre les différentes entrées utilisateurs.

J'utiliserai par la suite contact_doctor et contact_service pour designer les deux intentions.

## Librairies utilisées

### Rasa et Rasa X

Rasa est un framework d'apprentissage automatique open source pour les conversations textuelles et vocales automatisées. C'est sûrement l'outil le plus complet pour développer un chatbot.

J'ai choisi d'utiliser Rasa pour plusieurs raisons :
- Il permet grâce à Rasa X de facilement rendre disponible le bot à des utilisateurs tests.
- La plupart des composants utiles pour entraîner un chatbot sont disponibles. (DIETClassifier, RegexFeaturizer, ResponseSelector ...).
- Rasa permet facilement d'utiliser des composants provenant de framework extérieur comme spacy.
- La capacité d'entraîner des models basés sur du yml accélère grandement les choses.
- Si je suis pris pour ce poste, j'aurai eu l'occasion d'avoir une première expérience avec Rasa.

Points négatifs :
- J'ai eu un peu de mal à déployer rasa x sur mon serveur. (notamment le serveur d'action)
- C'est une usine à gaz. J'aurais pu utiliser une solution plus simple :
    - Spacy et faire de la classification avec comme interface un chat en ligne de commande.
    - Torch ou Tensorflow dans la même idée.
    - Les autres frameworks spécialisés en NLU (comme Snpis, Luis) sont très similaires à rasa ou proposent des solutions payantes (Saga).

https://rasa.com/docs/rasa/

### Spacy

Spacy est un framework python spécialisé dans le NLP. Open source et publié sous
licence MIT, il est sûrement l'outil le plus abouti pour faire de la NLP. La liste des fonctionnalités qu’il propose est vaste :

- Tokenization, lemmatization, Word Embedding
- Découpage en phrases
- Analyse Syntaxique (Tagger, Dependency Parsing, Attribute Ruler ...)
- Analyse Sémantique (NER, Entity Linking, Similarity)
- Solution Experte (Pattern Matching, Rule based Matching ...)

J'ai utilisé les composants spacy dans différentes pipelines.
J'ai aussi utilisé sa fonction de similarité (cosinus distance basé sur les vecteurs du word embbeding).

J'ai déjà eu l'occasion d'utiliser spacy durant ma période d'alternance l'année dernière.

https://spacy.io/usage

### Chatette

Chatette est un programme Python qui génère des ensembles de données spécialement pour entraîner des models de NLU.

Chatette fournit un DSL (Domain Specific Language) qui permet d'écrire des templates qui seront utilisés pour générer des phrases aléatoires.

J'ai fini par utiliser chatette dans la génération de phrases pour gagner du temps et de la lisibilité. C'est un outil très puissant que j'ai sans doute sous-exploité.

https://github.com/SimGus/Chatette

## Data

### Données utilisateurs

Durant toute la semaine, j'ai demandé à plusieurs personnes (amis, famille) de tester le bot via Rasa X. J'ai eu en tout une dizaine d'utilisateurs tests, âgés de 20 à 60 ans, certains issu du même parcours que moi (connaissance en code et en ML), d'autres pas du tout.

Leurs apports m'ont été très utiles pour avoir une idée plus générale du type de phrase possible pour exprimer les deux intentions. Ils m'ont aussi permis de constituer un petit jeu de test que j'ai utilisé tout au long du développement, au fur et mesure que celui-ci s'enrichissait.

Le jeu de test pose quand même quelques soucis :
- Il y a un peu moins de 200 phrases utilisables ce qui, pour moi, n'est pas suffisant pour évaluer correctement un modèle.
- Les utilisateurs ont beaucoup plus cherché a piéger le model qu'à le tester. C'est intéressant pour comprendre ses défauts, mais ça biaise l'évaluation. Un jeu de données réelles contiendrait, à mon sens, beaucoup plus de phrases classiques.
- Il est déséquilibré. J'ai eu beau donner à chacun des docteurs et services différents, certaines valeurs sont surreprésentées ou inexistantes.

### Première tentative de génération

Pour commencer, j'ai essayé de générer des phrases simples sans chatette par la permutation. Le résultat a donné des phrases génériques du type :

- "je veux téléphoner avec le docteur \[BENHAMOU\]\(doctor\)"
- "j'aurais aimé causer avec le médecin \[BRUNET Morgan\]\(doctor\)"
- "nous exigeons de parler à l'accueil du service \[rythmologie\]\(service\)"
- "il réclamerait de téléphoner avec le service \[chirurgie viscérale\]\(service\)"

Pour ce faire, j'ai scrap la conjugaison d'une dizaine de verbes (demander, pouvoir, vouloir ...) sur plusieurs temps (présent de l'indicatif, présent du conditionnel ...) et plusieurs pronoms (je, nous, il, ils). J'ai permuté ces verbes avec quelques morceaux de phrases types et la target (un service ou un docteur). J'ai pris aléatoire 10 000 phrases par (service, docteur) parmi les permutations.

J'ai fait quelques tests avec ce dataset et j'ai vite compris plusieurs problèmes aux utilisateurs externes (Je ne détaillerais pas ici la pipeline utilisée, j'en parle plus loin.) :
- Il faut spécifier une intention out_of_scope sinon le model cherche toujours à catégoriser la phrase comme contact_doctor ou contact_service. C'est assez logique et au final, c'est plus un oubli de ma part qu'un réel problème.
- De manière générale, le modèle avait du mal si l'objet du contact était avant la demande de contact. C'était tout à fait normal, aucune phrase dans le jeu d'entraînement ne présentait ce genre de configuration.
- Ex : "J'ai rendez vous avec le docteur X. Pouvez-vous me le passer ?"
- Lorsque plusieurs objets étaient présents dans une même phrase. Le modèle gardait le dernier trouvé comme target. La raison : je n'utilisais pas de rôle pour différencier les objets trouvés.
- Ex : "Je dois voir le docteur X sur recommandation du docteur Y." La target était forcément Y.

D'autres problèmes plus conceptuels :

Faut-il différencier la prise de rdv d'une demande de contact classique ? La question se pose aussi dans d'autres cas. Faut-il considérer que ces phrases sont des demandes de service / médecin ? (phrases prises dans le jeu de tests utilisateurs.)
- "j'aimerais un rdv chez l'opthtamo, j'ai mal !" -> demande du service : ophtalmologie
- "il faut absolument que je parle à l'assistante sociale" -> demande du service : Assistante sociale
- "Apparemment, il me faut une intervention en chirurgie viscérale" -> demande du service : Chirurgie Viscérale
- "mr brunet est-il disponible pour un rdv ?" -> demande du docteur : Brunet
- "le docteur benhamou doit me donner mes résultats" -> demande du docteur : Benhamou
- "Est-ce que le Dr Alonso est disponible ce matin ?" -> demande du docteur : Alonso

Dans le cadre où le bot aurait comme objectif d'être mis en production dans un hôpital ou une clinique, la question serait réglée avec l'aide du client. Dans mon cas actuel, j'ai décidé de choisir moi-même, au cas par cas. L'objectif étant de voir si le bot s'en sort bien pour détecter des demandes de contact plus implicites.

### Deuxième tentative de generation

Après la première tentative, j'ai décidé d'utiliser le module chatette.

J'ai gardé l'idée des verbes de la première génération. J'ai généré via python un fichier chatette contenant les verbes et les règles de générations. C'était un défi intéressant, mais une grosse perte de temps avec du recul, je pense que l'impact de cette grande variété de verbes sur les performances du modèle est minime voir négatif.

J'ai ajouté les possibilités :
- Qu'une phrase ait du bruit au début et la fin (bonjour, s'il vous plaît, c'est urgent)
- Que d'autres objets perturbent la phrase (à la demande du docteur X, qui travaille au service X ...).
- Qu'une phrase soit sous une forme interrogative (Puis-je joindre X ...). Jeu 1 seulement.
- Qu'il considère une prise de rdv comme une demande de contact. Jeu 1 seulement.
- Des phrases out_of_scope dont plusieurs sont générées à partir des bruits ajoutés aux autres phrases

J'ai généré le fichier main de chatette (celui qui contient les instructions nécéssaire à la génération) pour pouvoir avoir des intentions types : "contact_doctor/x". Cela m'a permis de tester le composant ResponseSelector de rasa. Dans cet objectif, j'ai des règles pour un contact_service/unknown et un contact_doctor/unknown qui ont pour rôle de gérer les demandes de services ou docteurs qui ne font pas partie des valeurs possibles.

Au final, je me retrouve avec deux jeux de données :

- Le jeu "0" :
    - Contienst 30 phrases par sous-intention.
    - 100 phrases pour contact_doctor/unknown et 100 phrases contact_service/unknown.
    - Soit 1000 phrases pour l'intention contact_doctor et l'intention contact_service.
    - 100 phrases out_of_scope.
    - Soit au total 2100 phrases.
- Le jeu "1" :
    - Contient 50 phrases par sous-intention.
    - 200 phrases pour contact_doctor/unknown et 200 phrases contact_service/unknown.
    - Soit 1700 phrases pour l'intention contact_doctor et l'intention contact_service.
    - 200 phrases out_of_scope.
    - Soit au total 3600 phrases.

### Conclusion sur les données

Quand on se renseigne sur la NLU et plus spécifiquement sur la conception de chatbot. On lit partout qu'il faut éviter les données générées et qu'il faut préférer des données réelles. À ce titre, j'ai sans doute passé trop de temps à générer des données et pas assez à en produire moi-même. Si je devais recommencer, je ferais une génération plus simple et plus contrôlée. 

La partie la plus compliquée de la génération, c'est l'intention out_of_scope. Il est compliqué d'appréhender et de fournir au modèle l'ensemble des cas qu'il doit considérer comme n'étant pas exactement une demande de contact. La plupart des erreurs du modèle se font avec cette intention.

Il reste sûrement des fautes d'orthographe / de frappe dans les templates chatette. De la même manière, certains patterns ne donnent pas des phrases parfaites.

- "bonsoir, ca va ? faut que tu m'aides ! je pourrais avoir un rdv avec Dr ABOAB" -> il manque un "le" entre "avec" et "Dr"
- "je souhaite échanger avec auprès du secrétariat du Admissions"

Je pense que ça ne pose pas de problème majeur. Le modèle est conçu pour gérer les fautes et tous les utilisateurs ne parleront pas un français parfait.

Pour finir sur ce sujet, j'ai choisi la taille des jeux données en me basant sur ce que j'ai trouvé sur Internet. J'ai commencé avec 30 phrases par sous-intention puis je me suis rendu compte que certaines intentions manquaient de contexte alors je suis passé à 50 pour voir si le model fonctionnait mieux.

## Pipelines

### Entité 

Ma première approche a consistée a utilisé le composant DIET. Il est composé de deux têtes, une qui classifie les intentions, une autre qui extrait les entités.

L'idée générale :
- Définir si l'intention est :
    - contact_doctor
    - contact_service
    - out_of_scope
- Extraire les entités :
    - doctor
    - doctor.target
    - service
    - service.target
- Assigner un slot qui récupère la dernière entité.target
- Répondre en fonction de l'intention et de la valeur du slot

Dans les faits, il n'y a pas un seul slot target, mais bien deux slots doctor et service. Pour être plus clair, je parlerais du slot target pour faire référence à ces deux slots.

Les avantages :
- Le modèle est plus facilement généralisable. On peut s'attendre qu'il fonctionne bien en ajoutant des services / docteurs dans sa base de connaissances.
- Les données d'entraînement sont plus faciles à manipuler. (seulement 3 intentions)

Les inconvénients :
- La target peut-être complètement hors sujet.
- Il faut d'une manière ou d'une autre traiter la target pour la faire correspondre exactement à la base de connaissance.

https://rasa.com/docs/rasa/components#dietclassifier

### Similarité entre les entités

Pour pouvoir utiliser efficacement cette première méthode, j'ai implémenté deux "custom action".

Elles ont pour rôle de récupérer le ou les docteurs / services proches de la valeur du slot target.

J'ai testé plusieurs méthodes de similarités (spacy cosinus distance, levenshtein, jaro, jaro-winkler). Pour les évaluer, j'ai effectué un ranking sur le jeu de test utilisateurs. Pour cela, j'ai profité de l'organisation du jeu de test en sous-intentions (contact_docor/x). J'ai donc utilisé chaque méthode, sur chaque exemple et récupérer la position de l'entité par rapport aux autres possibilités.

Exemple :

    - "Je cherche à joindre le Dr \[Aboab\]{"entity": "doctor", "role": "target"} pour prendre un rendez-vous."

    - L'intention pour cette phrase est : contact_doctor/aboab_jennifer
    - La valeur du slot target : Aboab
    - Pour chaque méthode :
        - Pour chaque médecin :
            - Je récupère le ratio de similarité entre les deux.
            - Je trie les résultats.
            - Je récupère le ranking de méthode en cherchant Aboab Jennifer dans les résultats triés.
            - Le meilleurs rank est donc 0 et le moins bon len(doctors) - 1

Les méthodes de similarités les plus performantes :

- Service : Levenshtein
    - rank :
        - mean : 0.03
        - min : 0.0
        - max : 2.0
    - ratio :
        - mean : 0.91
        - min : 0.5
        - max : 1.0
- Doctor : Jaro
    - rank :
        - mean : 0.2
        - min : 0.0
        - max : 9.0
    - ratio :
        - mean : 0.81
        - min : 0.5
        - max : 1.0

On peut voir qu'utiliser la similarité marche plutôt bien, surtout sur les services.

Les actions considèrent comme target :

- En priorité, n'importe quelle valeur qui a comme similarité 1 (càd le même mot).
- En second, toutes les valeurs dont le ratio de similarité est supérieur a un paramètre treshold (en pratique, treshold = .75)
- Et enfin si aucune valeur n'est trouvée à ce stade, les n premières valeurs (en pratique, n = 5)

Elles incluent à la réponse du bot, la ou les valeurs trouvées durant le processus.

https://rasa.com/docs/rasa/custom-actions/
https://en.wikipedia.org/wiki/Levenshtein_distance
https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
https://en.wikipedia.org/wiki/Cosine_similarity

### Response Selector

L'autre approche est basée sur le composant ResponseSelector. Je me suis directement inspiré de la manière dont sont traités les FAQ par les chatbots. Pour faire simple, on définit une intention principale a laquelle on ajoute une liste de sous-intentions. Dans notre cas, la liste des docteurs comme sous-intentions de contact_doctor. Pareil pour les services.

Les avantages :
- Pas besoin de s'embêter avec de la similarité ou autre pour faire correspondre la target à une valeur exacte.
- Il est plus facile de catégoriser des phrases que d'annoter des entités. L'annotation amène toujours à des ambiguïtés et se doit d'être guidée par une guideline stricte pour avoir un modèle efficace.

Les inconvénients :
- Le modèle est moins généralisable. Il faudra ajouter des exemples pour chaque nouvelle sous-intention.
- Le jeu d'entraînement est difficile à manipuler. Dans notre cas, il est composé de 62 sous-intentions.
- Contrairement à la similarité, on ne peut pas proposer les valeurs proches de la target si le modèle ne trouve pas exactement la target. Encore que, on pourrait ajouter un système de fallback qui proposerait les sous-entités avec les meilleurs scores si le model n'est pas sûr du résultat.

https://rasa.com/docs/rasa/components#responseselector

### Spacy

Rasa propose d'intégrer facilement des composants de spacy. Cela peut être utile pour utiliser un modèle pré entraîné ou un encoder plus élaborer.

Pour ma part, j'ai fait des tests avec :
- https://spacy.io/models/fr#fr_core_news_md
- https://spacy.io/models/fr#fr_dep_news_trf

Une des options possibles est d'utiliser le NER des pipelines spacy pour extraire les entités PER (personne) et les utiliser comme doctor. Je n'ai pas utilisé cette méthode par manque de temps et parce que je n'ai aucune idée de comment se comporterait le modèle en utilisant le NER d'une pipeline spacy et en définissant le rôle target.

### Epochs et loss

La configuration de base de rasa programme chaque composant pour être entraîné sur 100 epochs.
Avec tensorboard on voit que la loss se stabilise au bout de 50 epochs à peu près.
J'ai laissé le nombre d'epoch a 100 parceque le training n'est pas trop long (45 min pour les trf).
Je suppose que rasa implémente un algorithme de non-régression dans ça boucle d'entraînement.

J'ai activé constrain_similarities pour tous les composants qui le nécessite comme conseillé par la doc de rasa.
Je n'ai pas touché aux autres paramètres.

### Conclusion sur les pipelines

J'ai aussi utilisé les composants classiques de rasa :
- Tokenizer (Whitespace ou Spacy)
- RegexFeaturizer (avec les valeurs exactes des docteurs et services comme lookup tables)
- LexicalSyntacticFeaturizer (celui de Spacy pour les pipelines Spacy)
- CountVectorsFeaturizer (dont un avec une configuration pour les fautes)

Il y a encore beaucoup de composants que je n'ai pas pu tester :
- Différents intent classifier ou NER.
- L'utilisation des synonymes pour les entités. J'ai considéré que cette partie devait être exclusive à la génération de phrases. Comme j'ai dit plus haut sur le sujet, j'aurais dû faire plus simple et donc utiliser la feature synonyme de rasa.
- Knowledge Base Actions, j'ai découvert ce composant à la fin, quand j'ai développé les actions de similarités. Je pense que ça entre de le cadre d'utilisation de ce bot et qu'il aurait été intéressant de mettre en place cette fonctionnalité.

Je n'ai pas eu le temps de tester l'approche une intention et/ou une entité. Elle consisterait à réunir docteur et service en une seul entité objet et/ou contact_service et contact_docteur en une seule intention contact. Je n'ai aucune idée de la faisabilité/viabilité de cette approche, mais il aurait été intéressant de s'y pencher.

https://rasa.com/docs/rasa/components

## Résultats

### Le code

#### Installation 

Les prérequis sont ceux de rasa càd python Python 3.7 ou 3.8.

```
$ pip install -r requirements.txt
```

#### Generation de phrases

La partie génération de phrases se trouve dans le notebook : sentences_generation.ipynb

Le code est fouilli et absolument pas commenté. Comme j'ai déjà expliqué plus haut, je ne suis pas sûr qu'elle vaille vraiment le coup d'être retravaillée. Il vaudrait mieux opter pour des templates chatette plus simple. À la rigueur, seule la partie qui génère le template main pour qu'il soit adapté au ResponseSelector est intéressante.

#### EntitySimilarity

L'objet en lui-même est dans le fichier : src/libs/entity_similarity.py

Les actions qui l'utilisent dans le fichier : actions/actions.py

Une autre option aurait été de développer cette fonctionnalité comme un composant de la pipeline rasa.
Mais cela me semblait trop compliqué dans le temps qui m'était imparti.

Au finaln la class aurait pu très bien être une fonction mais j'avais anticipé fonctionnalités que je n'ai pas implémenté.

Rien de plus à dire de dessus. Il est commenté et il y a des docstrings.

#### Similarity Evaluation

Le script se trouve dans le fichier src/similarity_evaluation.py

Il s'utilise comme ceci :

```
$ python src/similarity_evaluation.py

usage: evaluate_similarity [-h] [--model-name MODEL_NAME] [--lookups-path LOOKUPS_PATH] [--examples-path EXAMPLES_PATH] [--role ROLE] [--output-dir OUTPUT_DIR]
                           [--best-method-criteria BEST_METHOD_CRITERIA]

Evaluate 4 methods of similarity on nlu examples.

Similarity methods :
    - Levenshtein
    - Jaro
    - Jaro-winkler
    - Word Embedding Cosinus Distance with spacy

Use entities values from nlu examples as first string.
Use each value in lookup tables as second string.

Entities should have the same name than a lookup table to be use.
If --role is provided, only entity with this role will be use.

Intent of example will be use for compute ranking of entity. Example :
    intent 'contact_doctor/x' will be parsed as x
    slugify(entity) == x will be use for ranking

Store report result in ouput-dir/{REPORT_FILE_NAME}.
Print best method based on best-method-criteria (less is better)


optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME, -M MODEL_NAME
                        The spacy pipeline to load. (default: fr_core_news_md)
  --lookups-path LOOKUPS_PATH, -L LOOKUPS_PATH
                        The yaml file use to find lookup tables. (default: data/lookups.yml)
  --examples-path EXAMPLES_PATH, -E EXAMPLES_PATH
                        The yaml file use to find nlu examples. (default: tests/nlu.yml)
  --role ROLE, -R ROLE  Role use for filter entities. (default: target)
  --output-dir OUTPUT_DIR, -O OUTPUT_DIR
                        directory used for store result. (default: results/)
  --best-method-criteria BEST_METHOD_CRITERIA, -B BEST_METHOD_CRITERIA
                        value used for choose best method (['mean', 'min', 'max', 'all']) (default: mean)
```

À la base, je voulais sortir des graphiques en output pour mieux visualiser les résultats, mais les boxplots et barchart sont peu lisibles à cause des petites valeurs. Faire un histogramme pour chaque méthode aurait généré trop de fichiers.

Les résultats détaillés pour le jeu de test utilisateur sont trouvables à cet endroit : results/similarity_report.json

#### ResponseSelector to Entity

Le script se trouve dans le fichier src/rs_to_entity.py

Il s'utilise comme ceci :

```
$ python src/rs_to_entity.py

usage: rs_to_entity [-h] [--nlu-path NLU_PATH] [--output-path OUTPUT_PATH] [--fill-role]

Convert ResponseSelector oriented nlu data file to an Entity nlu data file.



optional arguments:
  -h, --help            show this help message and exit
  --nlu-path NLU_PATH, -N NLU_PATH
                        path to nlu data yaml file to convert. (default: data/nlu.yml)
  --output-path OUTPUT_PATH, -O OUTPUT_PATH
                        path to save conversion result. (default: data/nlu_converted.yml)
  --fill-role, -F       if given, set value 'empty' to role entities if no other role is provided
    - rs_to_entity
```

L'objectif de ce script est de simplifier les intentions en enlevant leurs 'intent_response_key'.

Je n'étais pas sûr qu'il soit nécessaire de simplifier les intentions quand on passe d'une pipeline à l'autre mais après avoir effectué quelques tests, il semblerait que si.

J'ai ajouté l'option --fill-role parce que je voulais tester si le model se comportait différemment en précisant un rôle quelconque aux entités qui ne sont pas des target. Je n'ai pas eu le temps de la mettre en pratique.

#### Autres 

Le fichier src/libs/helpers.py qui contient quelques fonctions generiques que j'utilise dans d'autres partie du code (docstring complète).

Le fichier train/train.sh, un script bash qui lance le train du modèle puis le test sur son jeu de test généré par chatette et sur le jeu de test utilisateur.

### Rasa X

J'ai installé rasa-x en mode développement. Je l'ai seulement utilisé pour partager le chatbot et corriger les prédictions faites sur les entrées utilisateurs.

Le lien de l'interface admin : https://4a8a-51-159-36-19.eu.ngrok.io

Les logins sont dans le mail.

Vous pouvez switch le modèle à tester via le panel, sachez juste que le switch peu prendre du temps ...
Mon serveur n'est pas fait pour heberger ce genre de service.

Le lien du chatbot : https://4a8a-51-159-36-19.eu.ngrok.io/guest/conversations/production/25ed46c996c74ae48558998a45f3f402

### Les modèles

#### Organisation

Les modèles sont téléchargable ici (ou sur rasa x directement) : https://drive.google.com/drive/folders/1Pure_qbgWAox0NwJLsbr8od8SToaOyNF?usp=sharing

Les données d'entraînement et de test, la config et le domain sont trouvable dans le dossier : train

Ils sont nommés comme ceci :

```
3intents-[jeu de donnée]-[type de pipeline][base de pipeline]
example :
3intents-0-rs-base -> le modèle est entraîné avec le jeu de donnée 0 et la pipeline ResponseSelector basée sur rasa

les valeurs possibles :
    - jeu de donnée : [0, 1]
    - type de pipeline : [rs, e] pour [ResponseSelector, Entity]
    - base de pipeline : [base, md, trf] pour [rasa, fr_core_news_md, fr_dep_news_trf]
```

Tous les modèles sont upload sur Rasa X. 

Cependant suite à des problèmes de déploiement avec le serveur actions, les pipelines de type entity ne fonctionnent pleinement qu'en local.

Pour les utiliser : 

```
# Dans une première console :

$ rasa run actions

# Dans une deuxième console :

$ rasa shell nom_du_model

# se contenter de la deuxième console pour un modèle de type rs
```

#### Le metrics

Les metrics par rasa pour évaluer la perfomance d'un modèle sont les suivantes.

-  Precision (P) : Capacité d’un modèle à éviter les faux positifs
-  Recall (R) : Capacité d’un modèle à détecter tous les positifs
-  F-score (F1) = Moyenne Harmonique de la P et du R

Ce sont les metrics classiques utilisées dans une classification. Il existe des metrics plus complètes pour évaluer un NER mais je ne les ai pas utilisées.

Chaque composant est évalué indépendamment. Cela donne :

|FIELD1            |Intent   |||Entity   |||Response |||
|------------------|---------|------|------|---------|------|------|---------|------|-------|
|                  |precision|recall|f1    |precision|recall|f1    |precision|recall|f1     |
|3intents-0-rs-base|*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-0-rs-news|*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-0-rs-trf |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-0-e-base |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-0-e-news |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-0-e-trf  |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-rs-base|*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-rs-news|*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-rs-trf |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-e-base |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-e-news |*        |*     |*     |*        |*     |*     |*        |*     |*      |
|3intents-1-e-trf  |*        |*     |*     |*        |*     |*     |*        |*     |*      |

J'ai mis ici les performances sur le jeu de test utilisateurs.

Les résulats sur le jeu de test généré par chatette sont tous très proches de 1 et présentent peu d'intérêt à mon sens. D'une part parce que la géneration de phrases présente de gros défauts, d'autre part parce que ce qui nous intéresse, ce sont les performances du modèle sur des données proches du réel.

Les résultats sont difficilement interpretables pour plusieurs raisons :
    - Le jeu de test contient seulement X phrase (X contact_doctor, X contact_service, X out_of_scope). C'est encore pire pour ce qui concerne le ResponseSelector, certains docteurs ne sont même pas représentés.
    - Les utilisateurs test ont principalement cherché à piéger le model ce qui n'aide pas à l'évaluer correctement
    - L'annotation d'entité et la classification des intentions par une seule personne (moi) est souvent biaisée ou même incorrecte.

On peut quand même constater ceci :


Les résultats complets pour chaque models sont trouvables dans les dossiers : train/*/*/results et train/*/*/user_results 

## Conclusion

Comme je l'ai déjà dit plusieurs fois, je me suis emmêlé les pinceaux avec la génération de phrases qui m'a pris beaucoup de temps pour des résultats médiocres.

Une des causes est sûrement ma mauvaise organisation. J'avais comme suivi seulement des notes papier et j'ai très peu planifié ce que je devais faire. J'avais des idées très larges, mais pas de route précise pour les réaliser.

Un des meilleurs témoins de ce manque d'organisation est l'historique github. Il y a un nombre trop important de commit à quelques heures du rendu ...

La qualité du code a aussi souffert. Certaine parties mériteraient d'être refactorisées, d'autres d'être simplement commentées. J'aurais voulu faire des tests unitaires pour tout ce qui est similarités. Je n'ai pas mis en place de linter, je comptais le faire à la fin mais j'ai manqué de temps.

Une des manières d'éviter ces problèmes aurait été d'échanger un peu plus avec vous. J'aurais dû vous envoyer les phrases ambiguës et vous demander si oui ou non, il fallait les considérer comme des demandes de contact. Cela aurait permis de mieux cadrer l'objectif.

De manière générale, j'ai voulu tester trop de choses en trop peu de temps. J'aurais dû me concentrer sur certains aspects et les réaliser plus proprement.

J'ai quand même pu mettre un pied dans la nlu et la conception de chatbots. J'ai appris beaucoup de choses lors du déploiement de rasa-x. Échanger tout du long avec des utilisateurs tests était bénéfique et au final, j'ai exploré pas mal de choses en une semaine.

Je concluerai en vous remerciant d'avoir lu jusqu'au bout et en précisant qu'il est possible que je finisse certaines parties après vendredi 06/05 mais que tous les ajouts seront mis sur une autres branch que la branch main pour ne pas tricher.