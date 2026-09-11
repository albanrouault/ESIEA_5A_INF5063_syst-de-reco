"""Préparation des données : séparation des avis en entraînement / test."""

import numpy as np
import pandas as pd


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


def parts_validation_croisee(ratings, n_parts, seuil_pertinent, seed):
    """Découpe les AVIS en n_parts, utilisateur par utilisateur, films aimés et non aimés répartis séparément.

    Renvoie n_parts couples (train, validation) : chaque avis sert une fois en validation.
    Sert à choisir les hyperparamètres sans toucher au test (exercice 4).
    """
    rng = np.random.default_rng(seed)
    aime = ratings["rating"] >= seuil_pertinent
    strate = ratings["userId"].astype(str) + "_" + aime.astype(str)
    melange = ratings.sample(frac=1, random_state=rng)                       # ordre aléatoire
    strate_m = strate.reindex(melange.index)
    rang = melange.groupby(strate_m).cumcount()                              # rang de l'avis dans sa strate
    depart = pd.Series(rng.integers(n_parts, size=strate.nunique()), index=strate.unique())  # part de départ tirée au sort par strate : les restes ne tombent pas toujours dans les mêmes parts
    part = ((rang + depart[strate_m].to_numpy()) % n_parts).reindex(ratings.index)
    return [(ratings[part != i].reset_index(drop=True), ratings[part == i].reset_index(drop=True))
            for i in range(n_parts)]
