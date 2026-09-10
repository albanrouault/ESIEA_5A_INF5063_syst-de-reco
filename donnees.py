"""Préparation des données : séparation des avis en entraînement / test."""

import numpy as np


def split_par_utilisateur(ratings, frac_test, seuil_pertinent, seed):
    """Sépare les AVIS en train/test, utilisateur par utilisateur.

    Pour chaque utilisateur, frac_test de ses avis partent en test, en stratifiant sur
    "film aimé (note >= seuil_pertinent) / non aimé" avec au moins un avis par strate.
    Chaque utilisateur garde ainsi la majorité de ses avis en train et a au moins un film aimé en test.
    Sert aussi à découper une validation dans train (exercice 4).
    """
    # Un seul générateur pour tous les groupes : réutiliser la même graine dans chaque
    # groupe ferait tirer les mêmes positions pour tous les groupes de même taille.
    rng = np.random.default_rng(seed)
    aime = ratings["rating"] >= seuil_pertinent
    strate = ratings["userId"].astype(str) + "_" + aime.astype(str)

    def tirer(groupe):
        n_test = max(1, round(len(groupe) * frac_test))   # au moins 1 avis par strate
        return groupe.sample(n=n_test, random_state=rng)

    test = ratings.groupby(strate, group_keys=False).apply(tirer, include_groups=False)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)
