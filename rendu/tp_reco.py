"""TP systèmes de recommandation (INF5063, ESIEA 5A) : tout le code du TP en un seul fichier.

Lancer : uv run python tp_reco.py
Les tableaux s'affichent dans le terminal, les figures sont enregistrées dans le dossier figures/.
Les mesures longues (validation croisée) sont relues depuis resultats/ si les fichiers existent.
Le code suit l'ordre des exercices du sujet. Les réponses rédigées sont dans le notebook et le rapport.
"""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : les figures vont dans un dossier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)

RANDOM_STATE = 42  # graine de tous les tirages aléatoires
# Les trois CSV du dataset (movies_metadata, ratings_small, links_small) : on cherche dans les dossiers habituels
DATA_DIR = next(
    (
        d
        for d in (Path("Ressources/dataset"), Path("dataset"), Path("data"), Path("."))
        if (d / "ratings_small.csv").exists()
    ),
    Path("."),
)
DOSSIER_FIGURES = Path("figures")
DOSSIER_FIGURES.mkdir(exist_ok=True)


def section(texte):
    """Affiche un titre de section dans le terminal."""
    print("\n" + "=" * 100 + "\n" + texte + "\n" + "=" * 100)


def montrer(objet):
    """Affiche un tableau (ou n'importe quoi) dans le terminal."""
    print(objet.to_string() if isinstance(objet, (pd.DataFrame, pd.Series)) else objet)


def figure(nom):
    """Enregistre la figure en cours dans figures/nom.png."""
    chemin = DOSSIER_FIGURES / f"{nom}.png"
    plt.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[figure enregistrée : {chemin}]")


# ======================================================================
# Outils du TP : séparation des données, modèles, métriques
# ======================================================================


# ----------------------------------------------------------------------
# Séparation des avis : train / test et parts de validation croisée
# ----------------------------------------------------------------------
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
        n_test = max(1, round(len(groupe) * frac_test))  # au moins 1 avis par strate
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
    melange = ratings.sample(frac=1, random_state=rng)  # ordre aléatoire
    strate_m = strate.reindex(melange.index)
    rang = melange.groupby(strate_m).cumcount()  # rang de l'avis dans sa strate
    depart = pd.Series(
        rng.integers(n_parts, size=strate.nunique()), index=strate.unique()
    )  # part de départ tirée au sort par strate : les restes ne tombent pas toujours dans les mêmes parts
    part = ((rang + depart[strate_m].to_numpy()) % n_parts).reindex(ratings.index)
    return [
        (ratings[part != i].reset_index(drop=True), ratings[part == i].reset_index(drop=True)) for i in range(n_parts)
    ]


# ----------------------------------------------------------------------
# Modèles : filtrage collaboratif user-based et item-based
# ----------------------------------------------------------------------
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
        norme[norme == 0] = 1.0  # utilisateur sans variation : similarité 0
        sim = (X0 @ X0.T) / (norme @ norme.T)
        np.fill_diagonal(sim, 0.0)  # on n'est pas son propre voisin
        sim = np.clip(
            sim, 0.0, None
        )  # goûts opposés : pas un voisin (n'existe qu'avec le centrage, sinon tout est >= 0)
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
        if user not in self.R.index:  # utilisateur inconnu : moyenne globale partout
            return pd.Series(self.moyenne_globale, index=self.R.columns)
        v = self.voisins(user)
        notes = self.X.loc[v.index]  # notes des voisins (brutes ou centrées)
        poids = notes.notna().mul(v.to_numpy(), axis=0)  # similarité si le voisin a vu le film, 0 sinon
        pred = notes.fillna(0.0).mul(v.to_numpy(), axis=0).sum() / poids.sum()
        if self.centrer:
            pred = pred + self.moyenne[user]  # on remet le niveau de l'utilisateur
        return pred.fillna(self.moyenne[user])  # aucun voisin ne l'a vu : moyenne de l'utilisateur

    def predire(self, user):
        """Note prédite pour chaque film, ramenée sur l'échelle 0,5 à 5."""
        return self.score(user).clip(0.5, 5.0)

    def recommander(self, user, n=10):
        """Les n films non vus en train avec le meilleur score (classement sur le score non borné)."""
        if user not in self.R.index:  # utilisateur inconnu : rien à recommander
            return pd.DataFrame(columns=["note prédite", "voisins"])
        score = self.score(user)
        vus = self.R.loc[user].notna()
        n_voisins = self.R.loc[self.voisins(user).index].notna().sum()  # voisins ayant vu chaque film
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
        notes = self.R.loc[user, self.films].dropna()  # films notés par l'utilisateur
        S = self.sim[notes.index].to_numpy()
        if S.shape[1] > self.k:  # ne garder que les k plus similaires par ligne
            kieme = -np.partition(-S, self.k - 1, axis=1)[:, self.k - 1 : self.k]
            S = np.where(S >= kieme, S, 0.0)
        return pd.DataFrame(S, index=self.films, columns=notes.index), notes

    def score(self, user):
        """Score de chaque film : moyenne des notes de l'utilisateur pondérée par la similarité (non borné)."""
        if user not in self.R.index:
            return pd.Series(self.moyenne_globale, index=self.films)
        S, notes = self._poids(user)
        pred = pd.Series((S.to_numpy() @ notes.to_numpy()) / S.sum(axis=1), index=self.films)
        return pred.fillna(self.moyenne[user])  # aucun film voisin : moyenne de l'utilisateur

    def predire(self, user):
        """Note prédite pour chaque film assez noté, ramenée sur l'échelle 0,5 à 5."""
        return self.score(user).clip(0.5, 5.0)

    def recommander(self, user, n=10):
        """Algorithme du cours : n_favoris films préférés, k plus proches non vus de chacun, classement, top n."""
        if user not in self.R.index:
            return pd.DataFrame(columns=["note prédite", "voisins"])
        notes = self.R.loc[user, self.films].dropna()
        favoris = notes.nlargest(self.n_favoris)  # ses films préférés
        if favoris.empty:
            return pd.DataFrame(columns=["note prédite", "voisins"])
        S = self.sim[favoris.index].copy()  # similarité de chaque film avec les favoris
        S.loc[notes.index] = 0.0  # films déjà vus : pas candidats
        kieme = -np.partition(-S.to_numpy(), self.k - 1, axis=0)[self.k - 1]  # k-ième plus proche de chaque favori
        S = S.where(S >= kieme, 0.0)  # ne garder que les k plus proches par favori
        somme = pd.Series(S.to_numpy() @ favoris.to_numpy(), index=S.index)  # somme des similarités x notes des favoris
        note = somme / S.sum(axis=1)  # moyenne pondérée = note prédite
        score = somme if self.classement == "somme" else note
        top = score[S.sum(axis=1) > 0].nlargest(n)
        return pd.DataFrame(
            {"note prédite": note[top.index].clip(0.5, 5.0), "voisins": (S.loc[top.index] > 0).sum(axis=1)}
        )


# ----------------------------------------------------------------------
# Évaluation : RMSE / MAE, métriques top k, mesures d'un modèle
# ----------------------------------------------------------------------
def predire_test(reco, test):
    """Note prédite pour chaque avis de test, dans l'ordre de test."""
    preds = pd.Series(np.nan, index=test.index)
    for user, avis in test.groupby("userId"):
        pred = reco.predire(user).reindex(avis["movieId"])  # NaN si le film est inconnu du modèle
        preds[avis.index] = pred.fillna(reco.niveau(user)).to_numpy()  # film inconnu : moyenne de l'utilisateur
    return preds


def rmse_mae(y_vrai, y_pred):
    """RMSE et MAE entre notes vraies et notes prédites."""
    erreur = np.asarray(y_vrai) - np.asarray(y_pred)
    return np.sqrt(np.mean(erreur**2)), np.mean(np.abs(erreur))


def pertinents_test(test, seuil):
    """Films aimés (note >= seuil) de chaque utilisateur dans le test : {userId: ensemble de movieId}."""
    aimes = test[test["rating"] >= seuil]
    return aimes.groupby("userId")["movieId"].apply(set).to_dict()


def precision_rappel_ndcg(recommandes, pertinents, k=10):
    """Précision, rappel et NDCG top k pour un utilisateur.

    recommandes : movieId proposés, dans l'ordre ; pertinents : ensemble des movieId aimés cachés.
    La précision divise toujours par k : proposer moins de k films est pénalisé.
    """
    trouve = [film in pertinents for film in list(recommandes)[:k]]  # True si le film proposé est pertinent
    n_trouves = sum(trouve)
    precision = n_trouves / k
    rappel = n_trouves / len(pertinents) if pertinents else 0.0
    dcg = sum(1 / np.log2(pos + 2) for pos, ok in enumerate(trouve) if ok)  # gain décoté, position 0 = 1er
    idcg = sum(1 / np.log2(pos + 2) for pos in range(min(len(pertinents), k)))  # meilleur classement possible
    ndcg = dcg / idcg if idcg else 0.0
    return precision, rappel, ndcg


def evaluer_top_k(recommander, test, seuil, k=10):
    """Précision, rappel et NDCG top k de chaque utilisateur du test (une ligne par utilisateur).

    recommander(user, k) doit renvoyer les movieId proposés à user, dans l'ordre.
    Tous les utilisateurs du test sont évalués ; sans film aimé caché, les trois valent 0.
    """
    pertinents = pertinents_test(test, seuil)
    lignes = {
        user: precision_rappel_ndcg(recommander(user, k), pertinents.get(user, set()), k)
        for user in np.sort(test["userId"].unique())
    }
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
    popularite = np.median([n_avis.get(f, 0) for f in proposes]) if proposes else np.nan  # aucun film proposé : NaN
    distincts = len(set(proposes))
    return pd.Series(
        {"RMSE": rmse, "MAE": mae, **topk.to_dict(), "popularité médiane": popularite, "films distincts": distincts}
    )


def evaluer_populaire(train, valid, seuil, top_k=10):
    """Le repère sans modèle : à chaque utilisateur, les top_k films les plus notés en train qu'il n'a pas vus.

    Mêmes mesures qu'evaluer_modele (sans RMSE ni MAE : ce repère ne prédit pas de note).
    """
    n_avis = train.groupby("movieId").size().sort_values(ascending=False)
    vus = train.groupby("userId")["movieId"].apply(set)
    listes = {u: [f for f in n_avis.index if f not in vus.get(u, set())][:top_k] for u in valid["userId"].unique()}
    topk = evaluer_top_k(lambda u, k: listes[u][:k], valid, seuil, top_k).mean()
    proposes = [f for films in listes.values() for f in films]
    return pd.Series(
        {
            **topk.to_dict(),
            "popularité médiane": np.median([n_avis.get(f, 0) for f in proposes]),
            "films distincts": len(set(proposes)),
        }
    )


# ======================================================================
# Exercice 1 – Chargement des données
# ======================================================================
section("Exercice 1 – Chargement des données")


# --- 1) Chargement des fichiers ---
section("Exercice 1, 1) Chargement des fichiers")

movies = pd.read_csv(DATA_DIR / "movies_metadata.csv", low_memory=False)
ratings = pd.read_csv(DATA_DIR / "ratings_small.csv")
links = pd.read_csv(DATA_DIR / "links_small.csv")

tables = {"movies_metadata": movies, "ratings_small": ratings, "links_small": links}
for name, df in tables.items():
    print(f"{name:<16} {df.shape[0]:>7,} lignes  x {df.shape[1]:>2} colonnes")


# --- 2) Aperçu des tables ---
section("Exercice 1, 2) Aperçu des tables")

for name, df in tables.items():
    print(f"── {name} : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    montrer(df.head(3))


def resume_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Type, remplissage et cardinalité de chaque colonne."""
    return pd.DataFrame(
        {
            "type": df.dtypes.astype(str),
            "% nuls": (df.isna().mean() * 100).round(1),
            "valeurs distinctes": df.nunique(),
            "exemple": df.iloc[0].astype(str).str.slice(0, 40),
        }
    )


montrer(resume_colonnes(movies))

print("Valeurs de rating :", sorted(ratings["rating"].unique().tolist()))
print(
    "Période des avis  :",
    pd.to_datetime(ratings["timestamp"].min(), unit="s").date(),
    "->",
    pd.to_datetime(ratings["timestamp"].max(), unit="s").date(),
)
print("Nuls dans ratings :", ratings.isna().sum().sum(), "| nuls dans links :", links.isna().sum().to_dict())


# --- 3) Effectifs ---
section("Exercice 1, 3) Effectifs")

n_users = ratings["userId"].nunique()
n_movies_rated = ratings["movieId"].nunique()
n_ratings = len(ratings)
avis_par_user = ratings.groupby("userId").size()

montrer(
    pd.DataFrame(
        {
            "valeur": [
                n_users,
                n_ratings,
                n_movies_rated,
                links["movieId"].nunique(),
                movies["id"].nunique(),
                f"{avis_par_user.min()} / {avis_par_user.median():.0f} / {avis_par_user.max()}",
                f"{n_ratings / (n_users * n_movies_rated):.2%}",
                f"{(ratings['rating'] >= 4).mean():.0%}",
                (ratings.groupby("movieId").size() == 1).sum(),
            ],
        },
        index=[
            "utilisateurs",
            "avis",
            "films notés",
            "films dans links_small",
            "films dans movies_metadata",
            "avis par utilisateur (min / médiane / max)",
            "densité de la matrice utilisateurs x films",
            "part des avis >= 4",
            "films notés une seule fois",
        ],
    )
)


# --- 4) Rôle de links_small.csv ---
section("Exercice 1, 4) Rôle de links_small.csv")

# Le film movieId = 1 dans ratings : quel est-il ?
tmdb_id = int(links.loc[links["movieId"] == 1, "tmdbId"].iloc[0])
print(f"ratings movieId = 1  ->  links tmdbId = {tmdb_id}  ->  movies_metadata :")
montrer(movies.loc[movies["id"] == str(tmdb_id), ["id", "imdb_id", "title", "release_date"]])

print("movieId de ratings absents de links :", (~ratings["movieId"].isin(links["movieId"])).sum())
print("Films de links sans tmdbId          :", links["tmdbId"].isna().sum())
print(
    "tmdbId de links absents de metadata :",
    (~links["tmdbId"].dropna().astype(int).astype(str).isin(movies["id"])).sum(),
)


# --- 5) Nettoyage de movies_metadata ---
section("Exercice 1, 5) Nettoyage de movies_metadata")

# 1. ID au format numérique : les valeurs non convertibles deviennent NaN
id_num = pd.to_numeric(movies["id"], errors="coerce")
print(f"Lignes dont l'id n'est pas numérique : {id_num.isna().sum()}")
montrer(movies.loc[id_num.isna(), ["adult", "budget", "id", "title", "release_date", "popularity", "revenue"]])

# 2. Retrait de ces lignes et conversion en entier
movies_clean = movies.loc[id_num.notna()].copy()
movies_clean["id"] = id_num.loc[id_num.notna()].astype(int)

# 3. Doublons sur l'id
doublons = movies_clean[movies_clean.duplicated("id", keep=False)].sort_values("id")
print(
    f"Lignes en doublon sur l'id : {doublons['id'].duplicated().sum()} "
    f"({doublons['id'].nunique()} ids concernés, {doublons.duplicated().sum()} doublons strictement identiques)"
)

# Deux exemples : un doublon identique (105045) et un doublon qui diffère (4912)
montrer(
    doublons.loc[
        doublons["id"].isin([105045, 4912]), ["id", "title", "release_date", "popularity", "vote_count", "revenue"]
    ]
)

cols_diff = {c for _, g in doublons.groupby("id") for c in g.columns if g[c].astype(str).nunique() > 1}
print("Colonnes qui diffèrent entre doublons :", cols_diff)

movies_clean = movies_clean.drop_duplicates("id", keep="first").reset_index(drop=True)

print(f"Avant : {len(movies):,} lignes  ->  après : {len(movies_clean):,} lignes")
print("id unique :", movies_clean["id"].is_unique, "| dtype :", movies_clean["id"].dtype)


# --- 6) Nettoyage de ratings_small ---
section("Exercice 1, 6) Nettoyage de ratings_small")

n_avant = len(ratings)
ratings_clean = ratings.copy()

# 1. Avis manquants
manquants = ratings_clean[["userId", "movieId", "rating"]].isna().any(axis=1)
print(f"Avis avec valeur manquante          : {manquants.sum()}")
ratings_clean = ratings_clean[~manquants]

# 2. Identifiants conformes -> entiers
for col in ["userId", "movieId"]:
    num = pd.to_numeric(ratings_clean[col], errors="coerce")
    conforme = num.notna() & (num > 0) & (num % 1 == 0)
    print(f"{col:<8} non conformes               : {(~conforme).sum()}")
    ratings_clean = ratings_clean[conforme].assign(**{col: num[conforme].astype(int)})

# 3. Notes conformes
note_ok = ratings_clean["rating"].between(0.5, 5) & ((ratings_clean["rating"] * 2) % 1 == 0)
print(f"Notes hors échelle MovieLens        : {(~note_ok).sum()}")
ratings_clean = ratings_clean[note_ok]

# 4. Doublons (utilisateur, film)
dup = ratings_clean.duplicated(["userId", "movieId"])
print(f"Doublons (userId, movieId)          : {dup.sum()}")
ratings_clean = ratings_clean[~dup].reset_index(drop=True)

# Films sans métadonnées (conservés : le filtrage collaboratif n'en a pas besoin)
sans_meta = ~ratings_clean["movieId"].isin(links.loc[links["tmdbId"].isin(movies_clean["id"]), "movieId"])
print(f"Avis sur un film sans métadonnées   : {sans_meta.sum()} (conservés)")

print(f"\nAvant : {n_avant:,} avis  ->  après : {len(ratings_clean):,} avis")
print(ratings_clean.dtypes.to_string())


# --- 7) Les 10 films les plus anciens ---
section("Exercice 1, 7) Les 10 films les plus anciens")

# Conversion de release_date en date ; les valeurs invalides deviennent NaT
movies_clean["release_date"] = pd.to_datetime(movies_clean["release_date"], errors="coerce")

print("Films sans date de sortie :", movies_clean["release_date"].isna().sum())
futur = movies_clean[movies_clean["release_date"] > "2017-12-31"]
print(f"Films datés après 2017 (extraction du dataset) : {len(futur)}")
montrer(futur[["id", "title", "release_date", "status", "vote_count"]])

cols = ["id", "title", "release_date", "runtime", "original_language", "vote_count"]
montrer(movies_clean.dropna(subset=["release_date"]).nsmallest(10, "release_date")[cols])


# --- 8) Les 10 plus gros succès au box-office ---
section("Exercice 1, 8) Les 10 plus gros succès au box-office")

movies_clean["revenue"] = pd.to_numeric(movies_clean["revenue"], errors="coerce")
rev = movies_clean["revenue"]

print(f"revenue manquant (NaN)  : {rev.isna().sum():>6,}")
print(f"revenue = 0             : {(rev == 0).sum():>6,}  ({(rev == 0).mean():.1%} des films)")
print(f"0 < revenue < 1 000     : {rev.between(1, 999).sum():>6,}")
print(f"revenue >= 1 000        : {(rev >= 1000).sum():>6,}")

# Exemples de valeurs suspectes : quelques dollars de recettes pour des films sortis en salle
montrer(movies_clean.loc[rev.between(1, 999), ["title", "release_date", "budget", "revenue"]].head(5))

top_revenue = movies_clean.nlargest(10, "revenue").copy()
top_revenue["revenue (M$)"] = (top_revenue["revenue"] / 1e6).round(0)
top_revenue["budget (M$)"] = (pd.to_numeric(top_revenue["budget"], errors="coerce") / 1e6).round(0)
montrer(top_revenue[["id", "title", "release_date", "revenue (M$)", "budget (M$)", "vote_average", "vote_count"]])


# --- 9) Séparation entraînement / test ---
section("Exercice 1, 9) Séparation entraînement / test")

SEUIL_PERTINENT = 4.0  # un film noté >= 4 est considéré comme "aimé" (servira aussi pour les métriques top k)
FRAC_TEST = 0.2


train, test = split_par_utilisateur(ratings_clean, FRAC_TEST, SEUIL_PERTINENT, RANDOM_STATE)
print(f"train : {len(train):>6,} avis ({len(train) / len(ratings_clean):.1%})")
print(f"test  : {len(test):>6,} avis ({len(test) / len(ratings_clean):.1%})")

# Vérifications : chaque utilisateur est-il bien "testable" ?
par_user_train = train.groupby("userId").size()
par_user_test = test.groupby("userId").size()
aimes_test = test[test["rating"] >= SEUIL_PERTINENT].groupby("userId").size()

print(f"Utilisateurs en train / en test          : {train['userId'].nunique()} / {test['userId'].nunique()}")
print(f"Avis par utilisateur en train (min/méd.) : {par_user_train.min()} / {par_user_train.median():.0f}")
print(f"Avis par utilisateur en test  (min/méd.) : {par_user_test.min()} / {par_user_test.median():.0f}")
print(f"Utilisateurs sans film aimé en test      : {test['userId'].nunique() - len(aimes_test)}")
aimes_train = train[train["rating"] >= SEUIL_PERTINENT].groupby("userId").size()
print(
    f"Utilisateurs sans film aimé en train     : {train['userId'].nunique() - len(aimes_train)}",
    sorted(set(train["userId"]) - set(aimes_train.index)),
)
part_test = par_user_test / (par_user_train + par_user_test)
print(
    f"Part de test par utilisateur (min/méd/max): {part_test.min():.1%} / {part_test.median():.1%} / {part_test.max():.1%}"
)
print(f"Note moyenne train / test                : {train['rating'].mean():.3f} / {test['rating'].mean():.3f}")
print(f"Films aimés en test par utilisateur (méd.): {aimes_test.median():.0f}")
print("Avis communs train/test                  :", len(train.merge(test, on=["userId", "movieId"])))

absents = ~test["movieId"].isin(train["movieId"])
print(f"Avis de test sur un film absent du train : {absents.sum()} ({absents.mean():.1%})")

# Un exemple concret : l'utilisateur 221, 20 avis dont un seul film aimé
u = ratings_clean[ratings_clean["userId"] == 221].copy()
u["côté"] = np.where(u["movieId"].isin(test.loc[test["userId"] == 221, "movieId"]), "test", "train")
u["aimé"] = u["rating"] >= SEUIL_PERTINENT
print(u.groupby(["aimé", "côté"]).size().unstack(fill_value=0))
montrer(u.sort_values("rating", ascending=False)[["movieId", "rating", "aimé", "côté"]].head(6))


# ======================================================================
# Exercice 2 – Un bon système de filtrage collaboratif
# ======================================================================
section("Exercice 2 – Un bon système de filtrage collaboratif")


# --- 1) Mesure de similarité et nombre de voisins ---
section("Exercice 2, 1) Mesure de similarité et nombre de voisins")

# Sur combien de films deux utilisateurs peuvent-ils être comparés ? (train uniquement)
presence = train.pivot(index="userId", columns="movieId", values="rating").notna().astype(int)
communs = presence.to_numpy() @ presence.to_numpy().T  # films notés en commun, par paire
paires = communs[np.triu_indices_from(communs, k=1)]  # chaque paire une fois

print(
    f"Films en commun par paire : médiane {np.median(paires):.0f}, "
    f"{(paires < 5).mean():.0%} des paires en ont moins de 5"
)
voisins_fiables = (communs >= 5).sum(axis=1) - 1
print(f"Utilisateurs avec >= 5 films en commun, par utilisateur : médiane {np.median(voisins_fiables):.0f}")

K_VOISINS = 30  # nombre de voisins de départ, similarité cosinus


# --- 2) Matrice des avis et imputation ---
section("Exercice 2, 2) Matrice des avis et imputation")

# Matrice des avis (train) : une ligne par utilisateur, une colonne par film, NaN si non noté
R = train.pivot(index="userId", columns="movieId", values="rating")
print(
    f"{R.shape[0]} utilisateurs x {R.shape[1]} films, {R.notna().to_numpy().mean():.1%} de cases remplies, "
    f"{R.memory_usage().sum() / 1e6:.0f} Mo"
)

# Imputation : 0 pour les films non notés
R0 = R.fillna(0.0)
print("Cases vides restantes :", R0.isna().to_numpy().sum())

films_populaires = R.notna().sum().nlargest(8).index  # aperçu sur les 8 films les plus notés, avant / après
montrer(R[films_populaires].head(5))
montrer(R0[films_populaires].head(5))


# --- 3) Le système de filtrage collaboratif user-based ---
section("Exercice 2, 3) Le système de filtrage collaboratif user-based")

reco = RecoUserBased(k=K_VOISINS).fit(train)
print(f"Similarités : {reco.sim.shape}, min {reco.sim.to_numpy().min():.2f}, max {reco.sim.to_numpy().max():.2f}")

# Titre et année d'un movieId, via links (tmdbId) puis movies_clean (id)
titres = (
    links.dropna(subset=["tmdbId"])
    .astype({"tmdbId": int})
    .merge(movies_clean[["id", "title", "release_date"]], left_on="tmdbId", right_on="id")
    .set_index("movieId")
)
titres["titre"] = titres["title"] + " (" + titres["release_date"].dt.year.astype("Int64").astype(str) + ")"
titres = titres["titre"]


def avec_titres(df):
    """Ajoute le titre en première colonne d'un DataFrame indexé par movieId."""
    return df.assign(titre=titres.reindex(df.index).fillna("(titre inconnu)")).set_index("titre", append=True)


# Un utilisateur au hasard (générateur local, pour un tirage reproductible)
user = int(np.random.default_rng(RANDOM_STATE).choice(train["userId"].unique()))
print(f"Utilisateur {user} : {reco.R.loc[user].notna().sum()} films notés en train, moyenne {reco.moyenne[user]:.2f}")
print("Voisins :", reco.voisins(user).round(2).to_dict())

# Les voisins retenus partagent-ils assez de films avec lui ? (pour tous les utilisateurs)
co_notes = np.concatenate(
    [(reco.R.loc[reco.voisins(u).index].notna() & reco.R.loc[u].notna()).sum(axis=1).to_numpy() for u in reco.R.index]
)
print(
    f"Films en commun entre un utilisateur et ses voisins retenus : médiane {np.median(co_notes):.0f}, "
    f"{(co_notes < 5).mean():.0%} des voisins en ont moins de 5"
)

pred = reco.predire(user)
print(f"Notes prédites : {len(pred)} films, de {pred.min():.2f} à {pred.max():.2f}")
montrer(pred.head())

# Ses films préférés (train) et ses recommandations sans seuil, côte à côte
preferes = train[train["userId"] == user].set_index("movieId")["rating"].nlargest(10).to_frame("note")
print("Films préférés")
montrer(avec_titres(preferes))
print("Recommandations (min_voisins = 1)")
montrer(avec_titres(reco.recommander(user)))

# Le seuil ne joue que dans recommander : on compare 2, 3 et 5 sans ré-entraîner
for m in (2, 3, 5):
    reco.min_voisins = m
    r = reco.recommander(user).head(5)
    print(
        f"min_voisins = {m} :",
        " · ".join(f"{t} ({v} voisins)" for t, v in zip(titres.reindex(r.index).fillna("?"), r["voisins"])),
    )

MIN_VOISINS = 3  # voisins ayant vu un film pour pouvoir le recommander ; à optimiser à l'exercice 4

reco.min_voisins = MIN_VOISINS
print(f"Recommandations (min_voisins = {MIN_VOISINS})")
montrer(avec_titres(reco.recommander(user)).round(2))


# --- 4) Absence de fuite de données ---
section("Exercice 2, 4) Absence de fuite de données")

# 1. La matrice du modèle ne contient que les avis de train
print("Avis dans la matrice   :", reco.R.notna().to_numpy().sum(), "| avis en train :", len(train))
print("Films dans la matrice  :", reco.R.shape[1], "| films en train :", train["movieId"].nunique())

# 2. Aucun avis de test n'est dans la matrice : les cases (utilisateur, film) du test sont vides
test_connu = test[test["movieId"].isin(reco.R.columns)]
lig, col = reco.R.index.get_indexer(test_connu["userId"]), reco.R.columns.get_indexer(test_connu["movieId"])
assert (lig >= 0).all() and (col >= 0).all()  # -1 signalerait un utilisateur ou un film absent
print("Cases de test remplies :", np.isfinite(reco.R.to_numpy()[lig, col]).sum(), "sur", len(test_connu))

# 3. Les films connus seulement par le test sont inconnus du modèle
seulement_test = set(test["movieId"]) - set(train["movieId"])
print("Films seulement en test:", len(seulement_test), "| dans la matrice :", len(seulement_test & set(reco.R.columns)))

# 4. Le contrôle est sensible : un modèle entraîné sur train + test aurait d'autres similarités
sim_fuite = RecoUserBased(k=K_VOISINS).fit(pd.concat([train, test])).sim
print("Similarités identiques à un modèle entraîné avec le test :", np.allclose(sim_fuite, reco.sim))

# 5. La stratification du split sur la note change-t-elle le résultat ? Même modèle sur un tirage non stratifié
rng_s = np.random.default_rng(RANDOM_STATE)
test_ns = ratings_clean.groupby("userId").sample(frac=FRAC_TEST, random_state=rng_s)
train_ns = ratings_clean.drop(test_ns.index)
rmse_ns = rmse_mae(test_ns["rating"], predire_test(RecoUserBased(k=K_VOISINS).fit(train_ns), test_ns))[0]
print(
    f"RMSE du modèle : split stratifié {rmse_mae(test['rating'], predire_test(reco, test))[0]:.3f}, non stratifié {rmse_ns:.3f}"
)


# --- 5) RMSE et MAE sur l'ensemble de test ---
section("Exercice 2, 5) RMSE et MAE sur l'ensemble de test")

# Prédiction de chaque avis caché, puis RMSE et MAE
test_eval = test.copy()
test_eval["pred"] = predire_test(reco, test_eval)
rmse, mae = rmse_mae(test_eval["rating"], test_eval["pred"])
print(f"User-based (k = {K_VOISINS}) : RMSE = {rmse:.3f}   MAE = {mae:.3f}")

# Repères : que vaudrait un modèle sans voisins ?
moy_globale = train["rating"].mean()
moy_user = train.groupby("userId")["rating"].mean()
moy_film = train.groupby("movieId")["rating"].mean()

# Repère « biais » : moyenne globale + écart moyen du film + écart moyen de l'utilisateur (calculé sur le reste)
biais_film = (train["rating"] - moy_globale).groupby(train["movieId"]).mean()
biais_user = (
    (train["rating"] - moy_globale - biais_film.reindex(train["movieId"]).to_numpy()).groupby(train["userId"]).mean()
)
pred_biais = (
    moy_globale
    + biais_film.reindex(test_eval["movieId"]).fillna(0).to_numpy()
    + biais_user.reindex(test_eval["userId"]).fillna(0).to_numpy()
)

reperes = {
    "moyenne globale": np.full(len(test_eval), moy_globale),
    "moyenne de l'utilisateur": moy_user.reindex(test_eval["userId"]).to_numpy(),
    "moyenne du film": moy_film.reindex(test_eval["movieId"]).fillna(moy_globale).to_numpy(),
    "biais utilisateur + film": np.clip(pred_biais, 0.5, 5),
    f"user-based (k = {K_VOISINS})": test_eval["pred"].to_numpy(),
}
montrer(
    pd.DataFrame(
        [(nom, *rmse_mae(test_eval["rating"], p)) for nom, p in reperes.items()], columns=["modèle", "RMSE", "MAE"]
    )
    .set_index("modèle")
    .round(3)
)

# Où le modèle se trompe : films inconnus (repli) et erreurs selon la vraie note
inconnu = ~test_eval["movieId"].isin(reco.R.columns)
print(
    f"Avis sur film inconnu (repli sur la moyenne) : {inconnu.sum()}  RMSE = {rmse_mae(test_eval.loc[inconnu, 'rating'], test_eval.loc[inconnu, 'pred'])[0]:.3f}"
)
print(
    f"Avis sur film connu                          : {(~inconnu).sum()}  RMSE = {rmse_mae(test_eval.loc[~inconnu, 'rating'], test_eval.loc[~inconnu, 'pred'])[0]:.3f}"
)
test_eval["erreur"] = test_eval["pred"] - test_eval["rating"]
montrer(test_eval.groupby("rating")["erreur"].agg(["mean", "count"]).round(2).T)

# Deux causes mesurées : les prédictions sont tassées, et le niveau de l'utilisateur est ignoré
print(f"Écart-type des notes prédites / vraies : {test_eval['pred'].std():.2f} / {test_eval['rating'].std():.2f}")
niveau = moy_user.reindex(test_eval["userId"]).to_numpy()
print(
    f"Corrélation entre l'erreur (prédite - vraie) et le niveau de l'utilisateur : "
    f"{np.corrcoef(test_eval['erreur'], niveau)[0, 1]:.2f}"
)

# RMSE selon le nombre de voisins (parmi les k) qui ont vu le film
n_vois_avis = pd.Series(0, index=test_eval.index)
for u, avis in test_eval.groupby("userId"):
    vus = reco.R.loc[reco.voisins(u).index].notna().sum()  # voisins ayant vu chaque film
    n_vois_avis[avis.index] = vus.reindex(avis["movieId"]).fillna(0).to_numpy()
tranche = pd.cut(n_vois_avis, [-1, 0, 1, 2, 4, 9, 30], labels=["0", "1", "2", "3-4", "5-9", "10+"])
par_tranche = test_eval.groupby(tranche, observed=True)["erreur"]
montrer(
    pd.DataFrame({"RMSE": par_tranche.apply(lambda e: np.sqrt((e**2).mean())), "avis": par_tranche.size()}).round(3).T
)


# ======================================================================
# Exercice 3 – Évaluation
# ======================================================================
section("Exercice 3 – Évaluation")


# --- 1) Précision, rappel et NDCG top k ---
section("Exercice 3, 1) Précision, rappel et NDCG top k")


# --- 2) Implémentation des métriques ---
section("Exercice 3, 2) Implémentation des métriques")

# Test sur l'exemple de la question 1 : 8 films aimés cachés, 2 retrouvés en positions 1 et 4
aimes = {"Psycho", "Big Lebowski", "Fireflies", "Requiem", "Casino", "Big", "Desperado", "Pee-wee"}
liste = [
    "Psycho",
    "Pulp Fiction",
    "Chinatown",
    "Big Lebowski",
    "Annie Hall",
    "Eraserhead",
    "Lives of Others",
    "Happiness",
    "Miller's Crossing",
    "Get Shorty",
]
print("attendu 0.20 / 0.25 / 0.36  ->  obtenu", [round(x, 2) for x in precision_rappel_ndcg(liste, aimes)])


# --- 3) Précision, rappel et NDCG top 10 du système ---
section("Exercice 3, 3) Précision, rappel et NDCG top 10 du système")

TOP_K = 10  # taille de la liste recommandée et évaluée

# Notre système : les TOP_K films recommandés à chaque utilisateur (calculés une fois), puis les trois métriques
listes = {u: list(reco.recommander(u, TOP_K).index) for u in test["userId"].unique()}
topk = evaluer_top_k(lambda u, k: listes[u][:k], test, SEUIL_PERTINENT, k=TOP_K)
print(
    f"Utilisateurs avec au moins un film pertinent dans leur top {TOP_K} : {(topk.iloc[:, 0] > 0).sum()} / {len(topk)}"
)
montrer(topk.mean().round(3))

# Repères : les films les plus notés en train (non vus), et TOP_K films au hasard (non vus)
populaires = train.groupby("movieId").size().sort_values(ascending=False).index
vus_train = train.groupby("userId")["movieId"].apply(set)
rng_hasard = np.random.default_rng(RANDOM_STATE)
listes_pop = {u: [f for f in populaires if f not in vus_train[u]][:TOP_K] for u in listes}
listes_hasard = {
    u: list(rng_hasard.choice([f for f in populaires if f not in vus_train[u]], TOP_K, replace=False)) for u in listes
}

comparaison = pd.DataFrame(
    {
        "aléatoire": evaluer_top_k(lambda u, k: listes_hasard[u][:k], test, SEUIL_PERTINENT, k=TOP_K).mean(),
        "populaire": evaluer_top_k(lambda u, k: listes_pop[u][:k], test, SEUIL_PERTINENT, k=TOP_K).mean(),
        f"user-based (k = {K_VOISINS}, min_voisins = {MIN_VOISINS})": topk.mean(),
    }
).T.round(3)
montrer(comparaison)

# Où sont les bons films dans la liste, et quel rappel est atteignable ?
pertinents = pertinents_test(test, SEUIL_PERTINENT)


def positions_hits(listes):
    return [pos + 1 for u, films in pertinents.items() for pos, f in enumerate(listes[u]) if f in films]


pos_ub, pos_pop = positions_hits(listes), positions_hits(listes_pop)
print(
    f"Position moyenne des bons films dans le top {TOP_K} : user-based {np.mean(pos_ub):.1f} ({len(pos_ub)} bons films), "
    f"populaire {np.mean(pos_pop):.1f} ({len(pos_pop)} bons films)"
)

n_pert = np.array([len(f) for f in pertinents.values()])
print(
    f"Rappel@{TOP_K} maximal atteignable ({TOP_K} places par liste) : {np.mean(np.minimum(n_pert, TOP_K) / n_pert):.3f}"
)
n_absents = sum(len(f - set(train["movieId"])) for f in pertinents.values())
print(f"Films aimés cachés absents de train (jamais recommandables) : {n_absents} sur {n_pert.sum()}")

# Que recommande-t-on ? Popularité des films proposés, et recouvrement avec le « top des ventes »
n_avis = train.groupby("movieId").size()


def profil(listes):
    pop = np.median([n_avis.get(f, 0) for films in listes.values() for f in films])
    distincts = len({f for films in listes.values() for f in films})
    recouv = np.mean([len(set(films) & set(listes_pop[u])) for u, films in listes.items()])
    return pd.Series(
        {
            "popularité médiane (avis en train)": pop,
            "films distincts recommandés": distincts,
            f"films en commun avec le top populaire (sur {TOP_K})": round(recouv, 2),
        }
    )


print(f"Popularité médiane d'un film de train : {n_avis.median():.0f} avis")
montrer(pd.DataFrame({"user-based": profil(listes), "populaire": profil(listes_pop)}))


# ======================================================================
# Exercice 4 – Améliorations
# ======================================================================
section("Exercice 4 – Améliorations")


# --- 1) Optimisation des hyperparamètres ---
section("Exercice 4, 1) Optimisation des hyperparamètres")

# Grille sur les deux hyperparamètres, en validation croisée à 5 parts dans train (le test n'est pas touché)
GRILLE_K = [10, 20, 30, 50, 100]
GRILLE_MIN_VOISINS = [3, 5, 10, 15]
N_PARTS = 5
FICHIER_GRILLE = Path("resultats/grille_validation_croisee.csv")  # ~10 min de calcul : le résultat est gardé sur disque

if FICHIER_GRILLE.exists():
    grille = pd.read_csv(FICHIER_GRILLE)
else:
    parts = parts_validation_croisee(train, N_PARTS, SEUIL_PERTINENT, RANDOM_STATE)
    lignes = []
    for k in GRILLE_K:
        for m in GRILLE_MIN_VOISINS:
            for i, (train_fit, valid) in enumerate(parts):
                mesures = evaluer_modele(
                    RecoUserBased(k=k, min_voisins=m).fit(train_fit), valid, SEUIL_PERTINENT, TOP_K
                )
                lignes.append({"k": k, "min_voisins": m, "part": i, **mesures})
    grille = pd.DataFrame(lignes)
    FICHIER_GRILLE.parent.mkdir(exist_ok=True)
    grille.to_csv(FICHIER_GRILLE, index=False)
print(f"{len(grille)} lignes : {len(GRILLE_K)} valeurs de k x {len(GRILLE_MIN_VOISINS)} seuils x {N_PARTS} parts")

# Moyenne sur les 5 parts pour chaque réglage, et écart-type du NDCG entre parts (stabilité)
par_reglage = grille.groupby(["k", "min_voisins"])
resume = par_reglage.mean(numeric_only=True).drop(columns="part")
resume["NDCG écart-type"] = par_reglage["NDCG@10"].std()
montrer(resume.round(3))

# Lecture en deux tableaux : NDCG@10 et RMSE selon k (lignes) et min_voisins (colonnes)
print("NDCG@10 (moyenne des 5 parts)")
montrer(resume["NDCG@10"].unstack("min_voisins").round(3))
print("RMSE (moyenne des 5 parts)")
montrer(resume["RMSE"].unstack("min_voisins").round(3))

# La même chose en courbes, avec l'écart entre parts en barre d'erreur
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for m in GRILLE_MIN_VOISINS:
    r = resume.xs(m, level="min_voisins")
    axes[0].errorbar(
        r.index, r["NDCG@10"], yerr=r["NDCG écart-type"], marker="o", capsize=3, label=f"min_voisins = {m}"
    )
axes[1].plot(r.index, r["RMSE"], marker="o")  # une seule courbe : la RMSE ne dépend pas de min_voisins
axes[0].set(title="NDCG@10 en validation croisée", xlabel="k (voisins)", xscale="log")
axes[1].set(
    title="RMSE en validation croisée (même courbe pour tous les min_voisins)", xlabel="k (voisins)", xscale="log"
)
axes[0].legend()
plt.tight_layout()
figure("ex4_q1_courbes_validation")

# Prolongement de la crête « min_voisins ≈ k / 3 » au-delà de la grille, même validation croisée
CRETE = [(50, 20), (50, 25), (100, 25), (100, 35), (100, 50)]
FICHIER_CRETE = Path("resultats/crete_validation_croisee.csv")

if FICHIER_CRETE.exists():
    crete = pd.read_csv(FICHIER_CRETE)
else:
    parts = parts_validation_croisee(train, N_PARTS, SEUIL_PERTINENT, RANDOM_STATE)
    crete = pd.DataFrame(
        [
            {
                "k": k,
                "min_voisins": m,
                "part": i,
                **evaluer_modele(RecoUserBased(k=k, min_voisins=m).fit(train_fit), valid, SEUIL_PERTINENT, TOP_K),
            }
            for k, m in CRETE
            for i, (train_fit, valid) in enumerate(parts)
        ]
    )
    crete.to_csv(FICHIER_CRETE, index=False)

par_reglage = pd.concat([grille, crete]).groupby(["k", "min_voisins"])
resume = par_reglage.mean(numeric_only=True).drop(columns="part")
resume["NDCG écart-type"] = par_reglage["NDCG@10"].std()
montrer(resume.loc[[(30, 10), (50, 15)] + CRETE].round(3))

# Le compromis qualité / variété : un point par réglage, l'étoile est le réglage retenu
complement = pd.read_csv("resultats/v2_grille_complement_cv.csv")
resume = (
    pd.concat([grille, crete, complement]).groupby(["k", "min_voisins"]).mean(numeric_only=True).drop(columns="part")
)
r = resume[resume["films distincts"] > 0]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
sc = ax.scatter(r["films distincts"], r["NDCG@10"], c=r["popularité médiane"], cmap="viridis", s=60)
ax.scatter(
    *r.loc[[(30, 10)], ["films distincts", "NDCG@10"]].to_numpy().T,
    facecolor="none",
    edgecolor="#D62728",
    s=260,
    marker="*",
    linewidths=1.8,
    label="réglage retenu (30, 10)",
)
for k, m in [(50, 15), (30, 10), (25, 8), (100, 35), (100, 3), (50, 3), (100, 50), (10, 3)]:
    ax.annotate(
        f"({k}, {m})",
        (r.loc[(k, m), "films distincts"], r.loc[(k, m), "NDCG@10"]),
        fontsize=9,
        xytext=(6, 4),
        textcoords="offset points",
    )
plt.colorbar(sc, label="popularité médiane des films proposés (avis)")
ax.set(
    xlabel="films différents proposés (validation)",
    ylabel="NDCG@10 (validation)",
    title="Un point par réglage (k, min_voisins) : qualité du top 10 contre variété",
)
ax.legend(loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
figure("ex4_q1_qualite_vs_variete")

# Réglage retenu : règle à un écart-type, second critère = films distincts (voir la section Compléments)
complement = pd.read_csv("resultats/v2_grille_complement_cv.csv")
par = (
    pd.concat([grille, crete, complement])
    .groupby(["k", "min_voisins"])
    .agg(
        **{
            "NDCG@10": ("NDCG@10", "mean"),
            "NDCG écart-type": ("NDCG@10", "std"),
            "RMSE": ("RMSE", "mean"),
            "films distincts": ("films distincts", "mean"),
        }
    )
)
best = par["NDCG@10"].idxmax()
marge = par.loc[best, "NDCG écart-type"]
candidats = par[par["NDCG@10"] >= par.loc[best, "NDCG@10"] - marge]
meilleur_k, meilleur_min = candidats["films distincts"].idxmax()
print(
    f"Meilleur NDCG : {best} ({par.loc[best, 'NDCG@10']:.3f} ± {marge:.3f}) ; indépartageables : {list(candidats.index)}"
)
print(f"Réglage retenu (le plus varié parmi eux) : k = {meilleur_k}, min_voisins = {meilleur_min}")

reco_initial = RecoUserBased(k=K_VOISINS, min_voisins=MIN_VOISINS).fit(train)
reco_retenu = RecoUserBased(k=meilleur_k, min_voisins=meilleur_min).fit(train)
montrer(
    pd.DataFrame(
        {
            f"initial (k = {K_VOISINS}, min_voisins = {MIN_VOISINS})": evaluer_modele(
                reco_initial, test, SEUIL_PERTINENT, TOP_K
            ),
            f"retenu (k = {meilleur_k}, min_voisins = {meilleur_min})": evaluer_modele(
                reco_retenu, test, SEUIL_PERTINENT, TOP_K
            ),
            "(50, 15), maximum du NDCG": evaluer_modele(
                RecoUserBased(k=50, min_voisins=15).fit(train), test, SEUIL_PERTINENT, TOP_K
            ),
        }
    ).round(3)
)


# --- Compléments : d'autres réglages, et un seuil de similarité ---
section("Exercice 4, Compléments : d'autres réglages, et un seuil de similarité")

# Quatre réglages de plus sur la crête, à petit k
COMPLEMENT = [(20, 7), (20, 8), (25, 8), (25, 10)]
FICHIER_COMPLEMENT = Path("resultats/v2_grille_complement_cv.csv")
if FICHIER_COMPLEMENT.exists():
    complement = pd.read_csv(FICHIER_COMPLEMENT)
else:
    parts = parts_validation_croisee(train, N_PARTS, SEUIL_PERTINENT, RANDOM_STATE)
    complement = pd.DataFrame(
        [
            {
                "k": k,
                "min_voisins": m,
                "part": i,
                **evaluer_modele(RecoUserBased(k=k, min_voisins=m).fit(train_fit), valid, SEUIL_PERTINENT, TOP_K),
            }
            for k, m in COMPLEMENT
            for i, (train_fit, valid) in enumerate(parts)
        ]
    )
    complement.to_csv(FICHIER_COMPLEMENT, index=False)

par_reglage = pd.concat([grille, crete, complement]).groupby(["k", "min_voisins"])
resume = par_reglage.mean(numeric_only=True).drop(columns="part")
resume["NDCG écart-type"] = par_reglage["NDCG@10"].std()
montrer(
    resume.loc[
        [(20, 5), (20, 7), (20, 8), (20, 10), (25, 8), (25, 10), (30, 10), (50, 15)],
        ["RMSE", "NDCG@10", "NDCG écart-type", "précision@10", "rappel@10", "popularité médiane", "films distincts"],
    ].round(3)
)

# Écart apparié, part par part, entre (50, 15) et les réglages plus variés de la crête
g = pd.concat([grille, crete, complement]).set_index(["k", "min_voisins", "part"]).sort_index()
ref = g.loc[(50, 15)]
lignes = []
for k, m in [(30, 10), (25, 8), (20, 7)]:
    alt = g.loc[(k, m)]
    d = alt["NDCG@10"] - ref["NDCG@10"]
    lignes.append(
        {
            "réglage": f"({k}, {m})",
            "écart NDCG moyen": d.mean(),
            "parts où (50, 15) est meilleur": int((d < 0).sum()),
            "écart min / max": f"{d.min():+.3f} / {d.max():+.3f}",
            "films distincts en plus": f"{alt['films distincts'].mean() / ref['films distincts'].mean() - 1:+.0%}",
        }
    )
montrer(pd.DataFrame(lignes).set_index("réglage").round(3))

# Règle à un écart-type : indépartageables du meilleur, puis le plus varié
resume_tout = (
    pd.concat([grille, crete, complement])
    .groupby(["k", "min_voisins"])
    .agg(
        **{
            "NDCG@10": ("NDCG@10", "mean"),
            "NDCG écart-type": ("NDCG@10", "std"),
            "RMSE": ("RMSE", "mean"),
            "films distincts": ("films distincts", "mean"),
            "popularité médiane": ("popularité médiane", "mean"),
        }
    )
)
best = resume_tout["NDCG@10"].idxmax()
marge = resume_tout.loc[best, "NDCG écart-type"]
candidats = resume_tout[resume_tout["NDCG@10"] >= resume_tout.loc[best, "NDCG@10"] - marge].sort_values(
    "films distincts", ascending=False
)
print(
    f"Meilleur NDCG : {best}, {resume_tout.loc[best, 'NDCG@10']:.3f} ± {marge:.3f}  ->  indépartageables : {list(candidats.index)}"
)
print(f"Choix par la règle à un écart-type (le plus varié parmi eux) : {candidats.index[0]}")
montrer(candidats.round(3))


# Un seuil de similarité : en plus des k voisins, ou à la place (k = None)
class RecoSeuil(RecoUserBased):
    """RecoUserBased avec un seuil de similarité min_sim pour être voisin (le code de base a min_sim = 0)."""

    def __init__(self, min_sim=0.0, k=50, **kw):
        super().__init__(k=k if k else 10**6, **kw)
        self.min_sim = min_sim

    def voisins(self, user):
        v = self.sim.loc[user].nlargest(self.k)
        return v[v > self.min_sim]


SEUILS = {
    "référence (50, 15), seuil 0": dict(k=50, min_voisins=15, min_sim=0.0),
    "en plus : (50, 15) + seuil 0,10": dict(k=50, min_voisins=15, min_sim=0.10),
    "en plus : (50, 15) + seuil 0,15": dict(k=50, min_voisins=15, min_sim=0.15),
    "à la place : seuil 0,10, min_v 15": dict(k=None, min_voisins=15, min_sim=0.10),
    "à la place : seuil 0,15, min_v 15": dict(k=None, min_voisins=15, min_sim=0.15),
}
FICHIER_SEUIL = Path("resultats/v2_seuil_similarite_cv.csv")
if FICHIER_SEUIL.exists():
    seuil = pd.read_csv(FICHIER_SEUIL, index_col=0)
else:
    parts = parts_validation_croisee(train, N_PARTS, SEUIL_PERTINENT, RANDOM_STATE)
    seuil = pd.DataFrame(
        {
            nom: pd.concat(
                [evaluer_modele(RecoSeuil(**c).fit(t), v, SEUIL_PERTINENT, TOP_K) for t, v in parts], axis=1
            ).mean(axis=1)
            for nom, c in SEUILS.items()
        }
    ).T
    seuil.to_csv(FICHIER_SEUIL)
montrer(seuil[["RMSE", "NDCG@10", "précision@10", "rappel@10", "popularité médiane", "films distincts"]].round(3))

# Pourquoi le seuil seul échoue : les similarités sont très différentes d'un utilisateur à l'autre
S = reco_retenu.sim.to_numpy()
tri = -np.sort(-S, axis=1)
print(
    "similarité du 1er / 15e / 50e voisin (min / médiane / max selon l'utilisateur) :",
    " ; ".join(
        f"n°{p + 1} : {tri[:, p].min():.2f} / {np.median(tri[:, p]):.2f} / {tri[:, p].max():.2f}" for p in (0, 14, 49)
    ),
)
for s in (0.05, 0.10, 0.15, 0.20):
    n = (S > s).sum(axis=1)
    print(
        f"seuil {s:.2f} : de {n.min()} à {n.max()} voisins selon l'utilisateur (médiane {int(np.median(n))}), "
        f"{(n < meilleur_min).sum()} utilisateurs sous les {meilleur_min} voisins requis"
    )

# Vérification sur le test, une seule fois : le choix reste fait en validation
VERIF = [(50, 15), (30, 10), (25, 8), (20, 7)]
verif_test = pd.DataFrame(
    {
        f"({k}, {m})": evaluer_modele(RecoUserBased(k=k, min_voisins=m).fit(train), test, SEUIL_PERTINENT, TOP_K)
        for k, m in VERIF
    }
)
verif_test["films les plus notés"] = evaluer_populaire(train, test, SEUIL_PERTINENT, TOP_K)
montrer(verif_test.round(3))

# La variété en plus est-elle utile ? Les bons films trouvés (aimés cachés) sont-ils aussi plus variés ?
pert = pertinents_test(test, SEUIL_PERTINENT)
n_avis_train = train.groupby("movieId").size()
lignes = []
for k, m in VERIF:
    listes_km = {
        u: list(RecoUserBased(k=k, min_voisins=m).fit(train).recommander(u, TOP_K).index)
        for u in test["userId"].unique()
    }
    hits = [f for u, l in listes_km.items() for f in l if f in pert.get(u, set())]
    lignes.append(
        {
            "réglage": f"({k}, {m})",
            "films distincts proposés": len({f for l in listes_km.values() for f in l}),
            "bons films trouvés": len(hits),
            "bons films distincts": len(set(hits)),
            "popularité médiane des bons films": np.median([n_avis_train[f] for f in hits]),
        }
    )
montrer(pd.DataFrame(lignes).set_index("réglage"))


# --- 2) Une amélioration : le centrage des notes ---
section("Exercice 4, 2) Une amélioration : le centrage des notes")

# Brut et centré, même réglage (k = 30, min_voisins = 10) et mêmes 5 parts qu'à la question 1
parts = parts_validation_croisee(train, N_PARTS, SEUIL_PERTINENT, RANDOM_STATE)


def moyenne_validation(Modele, **reglages):
    mesures = [evaluer_modele(Modele(**reglages).fit(t), v, SEUIL_PERTINENT, TOP_K) for t, v in parts]
    return pd.concat(mesures, axis=1).mean(axis=1)


validation = pd.DataFrame(
    {
        "brut": moyenne_validation(RecoUserBased, k=meilleur_k, min_voisins=meilleur_min),
        "centré": moyenne_validation(RecoUserBased, k=meilleur_k, min_voisins=meilleur_min, centrer=True),
        "centré, seuil 7": moyenne_validation(
            RecoUserBased, k=meilleur_k, min_voisins=7, centrer=True
        ),  # le seuil a été réglé pour le brut
        "centré, seuil 13": moyenne_validation(RecoUserBased, k=meilleur_k, min_voisins=13, centrer=True),
        "films les plus notés": pd.concat(
            [evaluer_populaire(t, v, SEUIL_PERTINENT, TOP_K) for t, v in parts], axis=1
        ).mean(axis=1),  # repère sans modèle
    }
)
montrer(validation.round(3))

# Une seule fois sur le test : le système retenu (brut) et le même, centré
reco_centre = RecoUserBased(k=meilleur_k, min_voisins=meilleur_min, centrer=True).fit(train)
print(
    f"Paires d'utilisateurs de similarité > 0 : brut {(reco_retenu.sim > 0).mean().mean():.0%}, centré {(reco_centre.sim > 0).mean().mean():.0%}"
)
montrer(
    pd.DataFrame(
        {
            "brut": evaluer_modele(reco_retenu, test, SEUIL_PERTINENT, TOP_K),
            "centré": evaluer_modele(reco_centre, test, SEUIL_PERTINENT, TOP_K),
        }
    ).round(3)
)


# Ce que le centrage change dans les similarités : toutes les paires d'utilisateurs, avant le clip à 0
def similarites(X):
    X0 = X.fillna(0.0).to_numpy()
    norme = np.linalg.norm(X0, axis=1, keepdims=True)
    norme[norme == 0] = 1.0
    sim = (X0 @ X0.T) / (norme @ norme.T)
    return sim[np.triu_indices_from(sim, k=1)]  # chaque paire une fois


brut, centre = similarites(reco_retenu.X), similarites(reco_centre.X)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
bins = np.linspace(-0.3, 0.6, 46)
axes[0].hist(brut, bins=bins, color="#2F5BEA")
axes[0].set(
    title=f"Sans centrage : tout est positif (médiane {np.median(brut):.2f})",
    xlabel="similarité cosinus",
    ylabel="paires d'utilisateurs",
)
axes[1].hist(centre, bins=bins, color="#7A97FF")
axes[1].axvline(0, color="#444", ls="--", lw=1)
axes[1].set(
    title=f"Avec centrage : {(centre < 0).mean():.0%} des paires deviennent négatives", xlabel="similarité cosinus"
)
plt.tight_layout()
figure("ex4_q2_centrage_similarites")


# --- Bonus : un système item-based ---
section("Exercice 4, Bonus : un système item-based")

# Item-based en validation croisée (mêmes 5 parts) : les deux règles de classement, quelques réglages
REGLAGES_ITEM = {
    "note, min_avis 5, n 10, k 10": dict(classement="note", min_avis=5, n_favoris=10, k=10),
    "note, min_avis 50, n 10, k 10": dict(classement="note", min_avis=50, n_favoris=10, k=10),
    "somme, min_avis 5, n 20, k 20": dict(classement="somme", min_avis=5, n_favoris=20, k=20),
    "somme, min_avis 5, n 50, k 30": dict(classement="somme", min_avis=5, n_favoris=50, k=30),
    "somme, min_avis 20, n 50, k 30": dict(classement="somme", min_avis=20, n_favoris=50, k=30),
}
validation_item = pd.DataFrame({nom: moyenne_validation(RecoItemBased, **r) for nom, r in REGLAGES_ITEM.items()})
montrer(validation_item.round(3).T)

# Le repère, le système retenu et ses variantes, mesurés de la même façon en validation : NDCG, RMSE, popularité
colonnes = {
    "films les\nplus notés": validation["films les plus notés"],
    "user-based\nbrut (retenu)": validation["brut"],
    "user-based\ncentré": validation["centré"],
    "item-based\npar note": validation_item["note, min_avis 5, n 10, k 10"],
    "item-based\npar somme": validation_item[validation_item.loc["NDCG@10"].idxmax()],
}  # le meilleur réglage item-based
comp = pd.DataFrame(colonnes)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, ligne, titre in zip(
    axes,
    ["NDCG@10", "RMSE", "popularité médiane"],
    [
        "NDCG@10 (plus haut = mieux)",
        "RMSE (plus bas = mieux)",
        "Popularité médiane des films proposés (plus bas = moins de biais)",
    ],
):
    vals = comp.loc[ligne]
    barres = ax.bar(vals.index, vals.fillna(0), color=["#9aa5b1", "#2F5BEA", "#7A97FF", "#F2A43A", "#D97A0B"])
    ax.bar_label(
        barres,
        labels=[f"{v:.3f}" if ligne != "popularité médiane" else f"{v:.0f}" for v in vals.fillna(0)],
        padding=3,
        fontsize=9,
    )
    ax.set(title=titre)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25)
axes[1].set_ylim(0.85, 1.05)
axes[1].annotate("pas de note prédite", (0, 0.935), ha="center", fontsize=8.5, color="#666")
axes[2].axhline(3, color="#444", ls="--", lw=1)
axes[2].annotate("film médian du catalogue : 3 avis", (-0.4, 8), fontsize=8, color="#444")
plt.tight_layout()
figure("ex4_bonus_variantes_validation")

# Une seule fois sur le test : le réglage item-based au meilleur NDCG en validation, contre les deux user-based
nom_item = validation_item.loc["NDCG@10"].idxmax()
reco_item = RecoItemBased(**REGLAGES_ITEM[nom_item]).fit(train)
print(f"Item-based retenu : {nom_item} ({len(reco_item.films)} films avec une similarité)")
montrer(
    pd.DataFrame(
        {
            "user-based brut": evaluer_modele(reco_retenu, test, SEUIL_PERTINENT, TOP_K),
            "user-based centré": evaluer_modele(reco_centre, test, SEUIL_PERTINENT, TOP_K),
            "item-based": evaluer_modele(reco_item, test, SEUIL_PERTINENT, TOP_K),
        }
    ).round(3)
)


# ======================================================================
# Exercice 5 – Analyse critique
# ======================================================================
section("Exercice 5 – Analyse critique")

# Les top 10 de chaque système pour tous les utilisateurs, calculés une fois
utilisateurs = sorted(test["userId"].unique())
listes_ub = {u: list(reco_retenu.recommander(u, TOP_K).index) for u in utilisateurs}
listes_ib = {u: list(reco_item.recommander(u, TOP_K).index) for u in utilisateurs}
systemes = {"user-based retenu": listes_ub, "item-based (bonus)": listes_ib, "populaire": listes_pop}
print({nom: f"{sum(len(l) for l in listes.values())} films proposés" for nom, listes in systemes.items()})


# --- 1) Favorise-t-il les films populaires ? ---
section("Exercice 5, 1) Favorise-t-il les films populaires ?")

# Popularité des films proposés, comparée au catalogue et au top des ventes
top50 = set(populaires[:50])


def popularite(listes):
    films = [f for l in listes.values() for f in l]
    return pd.Series(
        {
            "popularité médiane (avis en train)": np.median([n_avis.get(f, 0) for f in films]),
            "part des recommandations dans le top 50": np.mean([f in top50 for f in films]),
            "part de films à moins de 20 avis": np.mean([n_avis.get(f, 0) < 20 for f in films]),
        }
    )


print(
    f"Catalogue : {len(n_avis)} films, popularité médiane {n_avis.median():.0f} avis, "
    f"{(n_avis < 20).mean():.0%} des films ont moins de 20 avis"
)
montrer(pd.DataFrame({nom: popularite(l) for nom, l in systemes.items()}).round(3))

# Répartition par tranche de popularité : le catalogue contre les films proposés
tranches, noms = [0, 2, 9, 49, 199, 10**9], ["1 à 2 avis", "3 à 9", "10 à 49", "50 à 199", "200 et plus"]


def par_tranche(valeurs):
    return pd.cut(pd.Series(valeurs), tranches, labels=noms).value_counts(normalize=True).reindex(noms).fillna(0) * 100


repartition = pd.DataFrame(
    {
        "catalogue": par_tranche(n_avis.to_numpy()),
        **{
            nom: par_tranche([n_avis.get(f, 0) for l in listes.values() for f in l]) for nom, listes in systemes.items()
        },
    }
)
ax = repartition.plot.bar(figsize=(9, 4), color=["#9aa5b1", "#2F5BEA", "#D97A0B", "#444"], rot=0, width=0.8)
ax.set(
    ylabel="part (%)",
    xlabel="popularité du film : nombre d'avis en train",
    title="Où se situent les films proposés par rapport au catalogue ?",
)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
figure("ex5_q1_popularite")


# --- 2) Les recommandations d'un utilisateur sont-elles diverses ? ---
section("Exercice 5, 2) Les recommandations d'un utilisateur sont-elles diverses ?")

# Genres de chaque film (movies_metadata via links), puis diversité de genres d'une liste
genres_tmdb = movies_clean.set_index("id")["genres"].map(
    lambda s: {g["name"] for g in ast.literal_eval(s)} if isinstance(s, str) else set()
)
genres_film = links.dropna(subset=["tmdbId"]).astype({"tmdbId": int}).set_index("movieId")["tmdbId"].map(genres_tmdb)


def diversite(films):
    """Genres distincts couverts par la liste, et part des paires de films sans genre commun."""
    g = [genres_film.get(f) if isinstance(genres_film.get(f), set) else set() for f in films]
    paires = [(a, b) for i, a in enumerate(g) for b in g[i + 1 :]]
    return len(set().union(*g)), np.mean([not (a & b) for a, b in paires]) if paires else np.nan


preferes = {u: list(train[train["userId"] == u].nlargest(TOP_K, "rating")["movieId"]) for u in utilisateurs}


def diversite_moyenne(listes):
    d = np.array([diversite(l) for l in listes.values() if l])
    return pd.Series({"genres distincts par liste": d[:, 0].mean(), "paires sans genre commun": np.nanmean(d[:, 1])})


montrer(
    pd.DataFrame(
        {
            **{nom: diversite_moyenne(l) for nom, l in systemes.items()},
            "films préférés de l'utilisateur": diversite_moyenne(preferes),
        }
    ).round(2)
)

# Exemple : l'utilisateur 60, ses préférés et son top 10 user-based, avec les genres
u = 60


def avec_genres(films):
    return pd.DataFrame(
        {
            "titre": titres.reindex(films).fillna("(titre inconnu)").to_numpy(),
            "genres": [", ".join(sorted(genres_film.get(f) or set())) for f in films],
        },
        index=films,
    )


print("Films préférés")
montrer(avec_genres(preferes[u]))
print("Top 10 user-based")
montrer(avec_genres(listes_ub[u]))

# La même chose sur les 671 utilisateurs : une boîte par type de liste (moitié centrale, trait = médiane)
groupes = {**systemes, "films préférés de l'utilisateur": preferes}
valeurs = {nom: np.array([diversite(l) for l in listes.values() if l]) for nom, listes in groupes.items()}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, j, titre in (
    (axes[0], 0, "Genres différents couverts par un top 10"),
    (axes[1], 1, "Part des paires de films sans genre commun"),
):
    ax.boxplot([v[:, j][~np.isnan(v[:, j])] for v in valeurs.values()], showfliers=False)
    ax.set(
        xticks=range(1, 5),
        xticklabels=[n.replace(" de l'utilisateur", "\nde l'utilisateur") for n in valeurs],
        title=titre,
    )
    ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
figure("ex5_q2_diversite_liste")


# --- 3) Les recommandations au global sont-elles diverses ? ---
section("Exercice 5, 3) Les recommandations au global sont-elles diverses ?")


# Couverture du catalogue et concentration des recommandations
def concentration(listes):
    comptes = pd.Series([f for l in listes.values() for f in l]).value_counts()
    cumul = comptes.cumsum() / comptes.sum()
    return pd.Series(
        {
            "films distincts proposés": len(comptes),
            "part du catalogue couverte": len(comptes) / len(n_avis),
            "films qui font la moitié des recommandations": int((cumul < 0.5).sum() + 1),
            "part prise par les 10 films les plus proposés": comptes.iloc[:10].sum() / comptes.sum(),
        }
    )


montrer(pd.DataFrame({nom: concentration(l) for nom, l in systemes.items()}).round(3))

# Les films les plus souvent proposés par le user-based, et à combien d'utilisateurs
plus_proposes = pd.Series([f for l in listes_ub.values() for f in l]).value_counts().head(5)
montrer(
    avec_titres(plus_proposes.to_frame("utilisateurs")).assign(
        avis_en_train=lambda d: n_avis.reindex(d.index.get_level_values(0)).to_numpy()
    )
)

# Les 15 films les plus proposés, et la part des utilisateurs qui les reçoivent ; puis la courbe de concentration
comptes_ub = pd.Series([f for l in listes_ub.values() for f in l]).value_counts()
comptes_ib = pd.Series([f for l in listes_ib.values() for f in l]).value_counts()
top15 = comptes_ub.head(15)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
y = np.arange(15)[::-1]
axes[0].barh(y + 0.2, top15.to_numpy() / len(utilisateurs) * 100, 0.4, color="#2F5BEA", label="user-based retenu")
axes[0].barh(
    y - 0.2,
    [comptes_ib.get(f, 0) / len(utilisateurs) * 100 for f in top15.index],
    0.4,
    color="#D97A0B",
    label="item-based, mêmes films",
)
axes[0].set(
    yticks=y,
    yticklabels=titres.reindex(top15.index).fillna("?"),
    xlabel="part des utilisateurs qui reçoivent ce film (%)",
    title="Les 15 films les plus proposés",
)
axes[0].legend(loc="lower right")
axes[0].tick_params(axis="y", labelsize=8)
for (nom, l), c in zip(systemes.items(), ["#2F5BEA", "#D97A0B", "#444"]):
    comptes = pd.Series([f for films in l.values() for f in films]).value_counts()
    cumul = comptes.cumsum() / comptes.sum()
    axes[1].plot(range(1, len(cumul) + 1), cumul * 100, color=c, lw=2, label=f"{nom} : {len(comptes)} films")
axes[1].axhline(50, color="#999", ls=":")
axes[1].set(
    xscale="log",
    xlabel="films, du plus proposé au moins proposé (log)",
    ylabel="part cumulée des recommandations (%)",
    title="Concentration",
)
axes[1].legend(fontsize=9)
plt.tight_layout()
figure("ex5_q3_plus_proposes_concentration")


# Par genre : part des films proposés qui portent chaque genre, comparée au catalogue
def part_genres(films):
    g = [genres_film.get(f) for f in films]
    g = [x for x in g if isinstance(x, set)]
    return pd.Series([genre for ens in g for genre in ens]).value_counts() / len(g) * 100


genres = pd.DataFrame(
    {
        "catalogue": part_genres(train["movieId"].unique()),
        **{
            nom: part_genres([f for l in listes.values() for f in l])
            for nom, listes in systemes.items()
            if nom != "populaire"
        },
    }
).fillna(0)
genres = genres.sort_values("catalogue", ascending=False).head(14)
ax = genres.plot.barh(figsize=(9, 5.5), color=["#9aa5b1", "#2F5BEA", "#D97A0B"], width=0.75)
ax.invert_yaxis()
ax.set(xlabel="part des films qui ont ce genre (%)", title="Quels genres sont proposés, comparés au catalogue ?")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
montrer(genres.round(1))
figure("ex5_q3_genres")


# --- 4) Les recommandations sont-elles réellement personnalisées ? ---
section("Exercice 5, 4) Les recommandations sont-elles réellement personnalisées ?")


# Films en commun entre deux listes (toutes les paires d'utilisateurs), et avec le top des ventes
def recouvrement(listes):
    ens = [set(l) for l in listes.values()]
    paires = [len(a & b) for i, a in enumerate(ens) for b in ens[i + 1 :]]
    avec_pop = [len(set(l) & set(listes_pop[u])) for u, l in listes.items()]
    return pd.Series(
        {
            "films en commun entre deux utilisateurs (moyenne)": np.mean(paires),
            "paires d'utilisateurs sans aucun film en commun": np.mean(np.array(paires) == 0),
            "films en commun avec le top des ventes": np.mean(avec_pop),
        }
    )


montrer(pd.DataFrame({nom: recouvrement(l) for nom, l in systemes.items()}).round(2))

# Deux utilisateurs aux goûts opposés : celui qui note le plus haut et celui qui note le plus bas les films d'action
action = genres_film[genres_film.map(lambda g: isinstance(g, set) and "Action" in g)].index
note_action = (
    train[train["movieId"].isin(action)].groupby("userId")["rating"].agg(["mean", "count"]).query("count >= 20")["mean"]
)
u_pour, u_contre = note_action.idxmax(), note_action.idxmin()
print(
    f"Utilisateur {u_pour} (note moyenne des films d'action {note_action[u_pour]:.2f}) et {u_contre} ({note_action[u_contre]:.2f}) : "
    f"{len(set(listes_ub[u_pour]) & set(listes_ub[u_contre]))} films en commun dans leurs top 10 user-based"
)
montrer(
    pd.DataFrame(
        {
            f"utilisateur {u_pour}": avec_genres(listes_ub[u_pour])["titre"].to_numpy(),
            f"utilisateur {u_contre}": avec_genres(listes_ub[u_contre])["titre"].to_numpy(),
        }
    )
)

# Pour chaque nombre de films en commun entre deux utilisateurs, la part des paires
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.27
for i, ((nom, l), c) in enumerate(zip(systemes.items(), ["#2F5BEA", "#D97A0B", "#444"])):
    ens = [set(x) for x in l.values()]
    communs = np.array([len(a & b) for j, a in enumerate(ens) for b in ens[j + 1 :]])
    ax.bar(
        np.arange(11) + (i - 1) * w,
        np.bincount(communs, minlength=11) / len(communs) * 100,
        w,
        color=c,
        label=f"{nom} : {communs.mean():.1f} en moyenne",
    )
ax.set(
    xticks=range(11),
    xlabel="films en commun entre les top 10 de deux utilisateurs",
    ylabel="part des paires (%)",
    title="Deux utilisateurs reçoivent-ils les mêmes films ?",
)
ax.legend()
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
figure("ex5_q4_personnalisation")


# --- 5) Comment améliorer le système en production ? ---
section("Exercice 5, 5) Comment améliorer le système en production ?")
