"""Modèles de filtrage collaboratif du TP."""

import numpy as np
import pandas as pd


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
        sim = np.clip(sim, 0.0, None)              # similarité négative = goûts opposés : pas un voisin
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

