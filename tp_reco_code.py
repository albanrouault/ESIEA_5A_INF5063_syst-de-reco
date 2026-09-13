"""TP systèmes de recommandation (INF5063) : tout le code du TP en un seul fichier.

Regroupe les trois modules utilisés par le notebook TP_reco.ipynb :
donnees (séparation des avis), modeles (RecoUserBased, RecoItemBased) et evaluation (métriques).
Le notebook contient les réponses aux questions ; ce fichier contient le code qu'il appelle.
"""

import numpy as np
import pandas as pd


# ============================================================================
# Préparation des données : séparation train / test et parts de validation croisée
# ============================================================================
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


# ============================================================================
# Modèles : filtrage collaboratif user-based et item-based
# ============================================================================
class RecoUserBased:
    """Filtrage collaboratif user-based : similarité cosinus, k voisins, moyenne pondérée.

    k           : nombre de voisins (au plus ; seuls les voisins de similarité > 0 comptent).
    min_voisins : voisins ayant vu un film pour pouvoir le recommander (1 = pas de seuil).
    centrer     : retirer à chaque utilisateur sa note moyenne avant de calculer les similarités.
                  Exercice 4.
    min_avis    : seuls les utilisateurs ayant au moins min_avis avis peuvent servir de voisins
                  (0 = tous). Tout le monde reste servi. Exercice 4.
    """

    def __init__(self, k=30, min_voisins=1, centrer=False, min_avis=0):
        self.k = k
        self.min_voisins = min_voisins
        self.centrer = centrer
        self.min_avis = min_avis

    def fit(self, train):
        # Matrice des avis (NaN si non noté) et note moyenne par utilisateur
        self.R = train.pivot(index="userId", columns="movieId", values="rating")
        self.moyenne = self.R.mean(axis=1)
        self.moyenne_globale = train["rating"].mean()

        # Notes brutes, ou centrées (note - moyenne de l'utilisateur) ; 0 dans les cases vides
        self.X = self.R.sub(self.moyenne, axis=0) if self.centrer else self.R
        X0 = self.X.fillna(0.0).to_numpy()

        # Similarité cosinus entre tous les utilisateurs, en un produit matriciel
        norme = np.linalg.norm(X0, axis=1, keepdims=True)
        norme[norme == 0] = 1.0                    # utilisateur sans variation : similarité 0
        sim = (X0 @ X0.T) / (norme @ norme.T)
        np.fill_diagonal(sim, 0.0)                 # on n'est pas son propre voisin
        sim = np.clip(sim, 0.0, None)              # goûts opposés : pas un voisin (n'existe qu'avec le centrage, sinon tout est >= 0)
        self.sim = pd.DataFrame(sim, index=self.R.index, columns=self.R.index)

        # Option : les utilisateurs trop peu actifs ne peuvent pas servir de voisins
        if self.min_avis:
            peu_actifs = self.R.notna().sum(axis=1) < self.min_avis
            self.sim.loc[:, peu_actifs[peu_actifs].index] = 0.0
        return self

    def niveau(self, user):
        """Note moyenne de l'utilisateur (moyenne globale s'il est inconnu du modèle)."""
        return self.moyenne.get(user, self.moyenne_globale)

    def voisins(self, user):
        """Les k utilisateurs les plus similaires (similarité > 0), avec leur similarité."""
        v = self.sim.loc[user].nlargest(self.k)
        return v[v > 0]

    def score(self, user):
        """Score de chaque film : moyenne des notes des voisins pondérée par leur similarité (non borné)."""
        if user not in self.R.index:               # utilisateur inconnu : moyenne globale partout
            return pd.Series(self.moyenne_globale, index=self.R.columns)
        v = self.voisins(user)
        notes = self.X.loc[v.index]                                        # notes des voisins (brutes ou centrées)
        poids = notes.notna().mul(v.to_numpy(), axis=0)                    # similarité si le voisin a vu le film, 0 sinon
        pred = notes.fillna(0.0).mul(v.to_numpy(), axis=0).sum() / poids.sum()
        if self.centrer:
            pred = pred + self.moyenne[user]                               # on remet le niveau de l'utilisateur
        return pred.fillna(self.moyenne[user])                             # aucun voisin ne l'a vu : moyenne de l'utilisateur

    def predire(self, user):
        """Note prédite pour chaque film, ramenée sur l'échelle 0,5 à 5."""
        return self.score(user).clip(0.5, 5.0)

    def recommander(self, user, n=10):
        """Les n films non vus en train avec le meilleur score (classement sur le score non borné)."""
        if user not in self.R.index:               # utilisateur inconnu : rien à recommander
            return pd.DataFrame(columns=["note prédite", "voisins"])
        score = self.score(user)
        vus = self.R.loc[user].notna()
        n_voisins = self.R.loc[self.voisins(user).index].notna().sum()   # voisins ayant vu chaque film
        top = score[~vus & (n_voisins >= self.min_voisins)].nlargest(n)
        return pd.DataFrame({"note prédite": top.clip(0.5, 5.0), "voisins": n_voisins[top.index]})



class RecoItemBased:
    """Filtrage collaboratif item-based : similarité cosinus entre films, moyenne pondérée des notes de l'utilisateur.

    Note prédite d'un film = moyenne des notes de l'utilisateur sur les k films notés les plus proches, pondérée par
    la similarité. Recommandation = algorithme du cours : ses n_favoris films préférés, les k plus proches non vus de
    chacun, puis classement. Exercice 4, bonus.

    k          : films voisins retenus (au plus) : par film à prédire, ou par favori pour recommander.
    n_favoris  : films préférés de l'utilisateur d'où partent les recommandations.
    min_avis   : un film doit avoir au moins min_avis avis en train pour avoir une similarité (sinon trop bruitée).
    classement : "note", par note prédite (cours) ; "somme", par somme des similarités x notes des favoris
                 (règle usuelle du top N item-based : un film relié à dix favoris passe devant un film relié à un seul).
    """

    def __init__(self, k=10, n_favoris=10, min_avis=5, classement="note"):
        self.k = k
        self.n_favoris = n_favoris
        self.min_avis = min_avis
        self.classement = classement

    def fit(self, train):
        self.R = train.pivot(index="userId", columns="movieId", values="rating")
        self.moyenne = self.R.mean(axis=1)
        self.moyenne_globale = train["rating"].mean()

        # Films assez notés ; 0 dans les cases vides, une ligne par film
        n_avis = self.R.notna().sum()
        self.films = n_avis[n_avis >= self.min_avis].index
        X0 = self.R[self.films].fillna(0.0).to_numpy().T

        # Similarité cosinus entre tous les films
        norme = np.linalg.norm(X0, axis=1, keepdims=True)
        sim = (X0 @ X0.T) / (norme @ norme.T)
        np.fill_diagonal(sim, 0.0)
        self.sim = pd.DataFrame(sim, index=self.films, columns=self.films)
        return self

    def niveau(self, user):
        """Note moyenne de l'utilisateur (moyenne globale s'il est inconnu du modèle)."""
        return self.moyenne.get(user, self.moyenne_globale)

    def _poids(self, user):
        """Similarité de chaque film (lignes) avec les k films les plus proches notés par user (colonnes), 0 ailleurs."""
        notes = self.R.loc[user, self.films].dropna()                       # films notés par l'utilisateur
        S = self.sim[notes.index].to_numpy()
        if S.shape[1] > self.k:                                            # ne garder que les k plus similaires par ligne
            kieme = -np.partition(-S, self.k - 1, axis=1)[:, self.k - 1:self.k]
            S = np.where(S >= kieme, S, 0.0)
        return pd.DataFrame(S, index=self.films, columns=notes.index), notes

    def score(self, user):
        """Score de chaque film : moyenne des notes de l'utilisateur pondérée par la similarité (non borné)."""
        if user not in self.R.index:
            return pd.Series(self.moyenne_globale, index=self.films)
        S, notes = self._poids(user)
        pred = pd.Series((S.to_numpy() @ notes.to_numpy()) / S.sum(axis=1), index=self.films)
        return pred.fillna(self.moyenne[user])                             # aucun film voisin : moyenne de l'utilisateur

    def predire(self, user):
        """Note prédite pour chaque film assez noté, ramenée sur l'échelle 0,5 à 5."""
        return self.score(user).clip(0.5, 5.0)

    def recommander(self, user, n=10):
        """Algorithme du cours : n_favoris films préférés, k plus proches non vus de chacun, classement, top n."""
        if user not in self.R.index:
            return pd.DataFrame(columns=["note prédite", "voisins"])
        notes = self.R.loc[user, self.films].dropna()
        favoris = notes.nlargest(self.n_favoris)                            # ses films préférés
        if favoris.empty:
            return pd.DataFrame(columns=["note prédite", "voisins"])
        S = self.sim[favoris.index].copy()                                  # similarité de chaque film avec les favoris
        S.loc[notes.index] = 0.0                                            # films déjà vus : pas candidats
        kieme = -np.partition(-S.to_numpy(), self.k - 1, axis=0)[self.k - 1]   # k-ième plus proche de chaque favori
        S = S.where(S >= kieme, 0.0)                                        # ne garder que les k plus proches par favori
        somme = pd.Series(S.to_numpy() @ favoris.to_numpy(), index=S.index)  # somme des similarités x notes des favoris
        note = somme / S.sum(axis=1)                                        # moyenne pondérée = note prédite
        score = somme if self.classement == "somme" else note
        top = score[S.sum(axis=1) > 0].nlargest(n)
        return pd.DataFrame({"note prédite": note[top.index].clip(0.5, 5.0), "voisins": (S.loc[top.index] > 0).sum(axis=1)})


# ============================================================================
# Évaluation : RMSE / MAE, métriques top k, mesures d'un modèle et du repère populaire
# ============================================================================
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


def evaluer_populaire(train, valid, seuil, top_k=10):
    """Le repère sans modèle : à chaque utilisateur, les top_k films les plus notés en train qu'il n'a pas vus.

    Mêmes mesures qu'evaluer_modele (sans RMSE ni MAE : ce repère ne prédit pas de note).
    """
    n_avis = train.groupby("movieId").size().sort_values(ascending=False)
    vus = train.groupby("userId")["movieId"].apply(set)
    listes = {u: [f for f in n_avis.index if f not in vus.get(u, set())][:top_k] for u in valid["userId"].unique()}
    topk = evaluer_top_k(lambda u, k: listes[u][:k], valid, seuil, top_k).mean()
    proposes = [f for films in listes.values() for f in films]
    return pd.Series({**topk.to_dict(), "popularité médiane": np.median([n_avis.get(f, 0) for f in proposes]),
                      "films distincts": len(set(proposes))})
