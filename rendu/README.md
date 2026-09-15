# TP systèmes de recommandation – rendu

Rambaud, Saget, Rouault – INF5063 (ESIEA 5A).

- `tp_reco.py` : tout le code du TP en un seul script, dans l'ordre des exercices. C'est le fichier déposé pour la question « script Python unique ».
- `TP_reco.ipynb` et `TP_reco_execute.html` : le notebook avec les réponses rédigées, et son export exécuté avec toutes les sorties.
- `dataset/` : les trois CSV utilisés (The Movies Dataset, Kaggle).
- `resultats/` : les mesures de la validation croisée déjà calculées. Si le dossier manque, le script les recalcule (environ 25 minutes de plus).

Lancer le script depuis ce dossier (testé avec Python 3.12, pandas 3.0, numpy 2.5, matplotlib 3.11 ; pandas 2.2 minimum) :

    python tp_reco.py

Les tableaux s'affichent dans le terminal, les figures sont enregistrées dans `figures/`. Durée : environ 10 minutes.
