"""Évaluation des modèles sur l'ensemble de test."""

import numpy as np
import pandas as pd


def predire_test(reco, test):
    """Note prédite pour chaque avis de test, dans l'ordre de test."""
    preds = pd.Series(np.nan, index=test.index)
    for user, avis in test.groupby("userId"):
        pred = reco.predire(user).reindex(avis["movieId"])          # NaN si le film est inconnu du modèle
        preds[avis.index] = pred.fillna(reco.niveau(user)).to_numpy()  # film inconnu : moyenne de l'utilisateur
    return preds


def rmse_mae(y_vrai, y_pred):
    """RMSE et MAE entre notes vraies et notes prédites."""
    erreur = np.asarray(y_vrai) - np.asarray(y_pred)
    return np.sqrt(np.mean(erreur ** 2)), np.mean(np.abs(erreur))
