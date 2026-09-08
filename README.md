# TP – Systèmes de recommandation

INF5063 – Machine learning : applications (ESIEA 5A).
Conception et évaluation d'un système de recommandation de films par filtrage collaboratif
sur [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

Sujet : `TP systèmes de recommandation.pdf`. Travail : `TP_reco.ipynb`.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/albanrouault/ESIEA_5A_INF5063_syst-de-reco.git
cd ESIEA_5A_INF5063_syst-de-reco
uv sync                            # crée .venv et installe les dépendances
uv run nbstripout --install        # à faire UNE FOIS par clone (voir ci-dessous)
```

### Données

Les données ne sont pas versionnées (900 Mo). Télécharger
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) sur Kaggle
et décompresser l'archive dans `Ressources/dataset/`. Le notebook n'utilise que :

```
Ressources/dataset/movies_metadata.csv
Ressources/dataset/ratings_small.csv
Ressources/dataset/links_small.csv
```

### Lancer le notebook

```bash
uv run jupyter lab
```

## Travailler à plusieurs sur le notebook

Les fichiers `.ipynb` stockent les sorties des cellules (tableaux, graphiques en base64) et
des métadonnées d'exécution. Sans précaution, chaque exécution modifie le fichier et deux
personnes qui touchent le même notebook obtiennent des conflits ingérables.

Le dépôt est donc configuré avec **nbstripout** (`.gitattributes`) : au moment du `git add`,
les sorties et compteurs d'exécution sont retirés de la version commitée. Le fichier local
garde ses sorties, seul le dépôt est propre.

Conséquences :

- `uv run nbstripout --install` doit être lancé **une fois par clone** (le filtre est stocké
  dans `.git/config`, qui n'est pas versionné). Sans ça, vos commits contiendraient les sorties.
- Après un `git pull`, les cellules apparaissent sans sortie : ré-exécuter le notebook.
- Pour comparer deux versions d'un notebook de façon lisible :
  `uv run nbdiff-web HEAD~1 HEAD TP_reco.ipynb` (ou `uv run nbdiff`).
- Bonnes pratiques : `git pull` avant de commencer, commits fréquents et ciblés,
  et si possible ne pas éditer la même question en même temps.

## Dépendances

Déclarées dans `pyproject.toml`, verrouillées dans `uv.lock`. Pour en ajouter une :
`uv add <paquet>` puis commiter `pyproject.toml` et `uv.lock`.
