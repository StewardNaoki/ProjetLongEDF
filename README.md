# ProjetLongEDF

## Les Dossiers
- *src/*: Contient le code python
- *log/*: Contient les logs des différents entraînenements
- *examples/*: Contient des exemples de résolution de problèmes LP avec **Pulp**
- *DATA/*: Contient Les différentes données

## Description des ficiers scources
Préparation et conversion de données:

- *convert_json_csv.py*: Conversion des .json en fichier .csv (Problème EDF)
- *generate_pb_csv.py*: generateur de problèmes LP (Problème LP)
- *Dataset*: Prépare le générateurs de données pour l'entraînement

Entrainement du réseau:
- *Network.py*: Fichier gérant l'entraînemenet du réseau.
- *main_EDF.py & main_LP.py*: fichier à lancer pour crées toutes les instances utilisées pour l'entraînement
- *loss_EDF.py & loss_LP.py*: définit les foonctions de pertes des 2 problèmes
- *log_writer.py*: Définit les fonctions de log.

## Les scripts
- *clear_log.sh*: Nettoie le dossier log
- *tensorboard_connect.sh*: connection tensorboard
- *launch.sh*: scrit à lancer pour lancer l'entraînement
