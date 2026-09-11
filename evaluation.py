"""Évaluation des modèles sur un ensemble d'avis cachés (test ou validation)."""

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


def pertinents_test(test, seuil):
    """Films aimés (note >= seuil) de chaque utilisateur dans le test : {userId: ensemble de movieId}."""
    aimes = test[test["rating"] >= seuil]
    return aimes.groupby("userId")["movieId"].apply(set).to_dict()


def precision_rappel_ndcg(recommandes, pertinents, k=10):
    """Précision, rappel et NDCG top k pour un utilisateur.

    recommandes : movieId proposés, dans l'ordre ; pertinents : ensemble des movieId aimés cachés.
    La précision divise toujours par k : proposer moins de k films est pénalisé.
    """
    trouve = [film in pertinents for film in list(recommandes)[:k]]     # True si le film proposé est pertinent
    n_trouves = sum(trouve)
    precision = n_trouves / k
    rappel = n_trouves / len(pertinents) if pertinents else 0.0
    dcg = sum(1 / np.log2(pos + 2) for pos, ok in enumerate(trouve) if ok)   # gain décoté, position 0 = 1er
    idcg = sum(1 / np.log2(pos + 2) for pos in range(min(len(pertinents), k)))  # meilleur classement possible
    ndcg = dcg / idcg if idcg else 0.0
    return precision, rappel, ndcg


def evaluer_top_k(recommander, test, seuil, k=10):
    """Précision, rappel et NDCG top k de chaque utilisateur du test (une ligne par utilisateur).

    recommander(user, k) doit renvoyer les movieId proposés à user, dans l'ordre.
    Tous les utilisateurs du test sont évalués ; sans film aimé caché, les trois valent 0.
    """
    pertinents = pertinents_test(test, seuil)
    lignes = {user: precision_rappel_ndcg(recommander(user, k), pertinents.get(user, set()), k)
              for user in np.sort(test["userId"].unique())}
    return pd.DataFrame.from_dict(lignes, orient="index", columns=[f"précision@{k}", f"rappel@{k}", f"NDCG@{k}"])


def evaluer_modele(reco, valid, seuil, top_k=10):
    """Toutes les mesures d'un modèle entraîné, sur un ensemble d'avis cachés (validation ou test).

    RMSE, MAE, précision, rappel et NDCG top k, popularité médiane des films proposés
    (nombre d'avis en train) et nombre de films distincts proposés.
    """
    rmse, mae = rmse_mae(valid["rating"], predire_test(reco, valid))
    listes = {u: list(reco.recommander(u, top_k).index) for u in valid["userId"].unique()}
    topk = evaluer_top_k(lambda u, k: listes[u][:k], valid, seuil, top_k).mean()
    n_avis = reco.R.notna().sum()
    proposes = [f for films in listes.values() for f in films]
    popularite = np.median([n_avis.get(f, 0) for f in proposes]) if proposes else np.nan   # aucun film proposé : NaN
    distincts = len(set(proposes))
    return pd.Series({"RMSE": rmse, "MAE": mae, **topk.to_dict(),
                      "popularité médiane": popularite, "films distincts": distincts})
