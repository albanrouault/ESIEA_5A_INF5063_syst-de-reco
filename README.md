# TP – Systèmes de recommandation

INF5063 – Machine learning : applications (ESIEA 5A).
Conception et évaluation d'un système de recommandation de films par filtrage collaboratif
sur [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

Sujet : `Ressources/TP systèmes de recommandation.pdf`. Cours : `Ressources/Systèmes de recommandations (1).pdf`.

Travail : `TP_reco.ipynb` (questions et réponses), `donnees.py` (séparation train/test), `modeles.py` (modèles de recommandation), `evaluation.py` (métriques).

Pour le rendu : `tp_reco_code.py` regroupe les trois modules en un seul fichier (même code, à régénérer si les modules changent), `TP_reco_execute.html` est l'export du notebook exécuté avec ses sorties, `resultats/` contient les mesures de la validation croisée (CSV), `questionnaire/v1/` les réponses au questionnaire Moodle (un Word par question, le rapport complet, les figures déposées dans `ressources/`), `questionnaire/v2/` un rapport alternatif complet où le système retenu est (30, 10) par la règle à un écart-type (Word, figures dans `ressources/`), recalculé et rédigé dans `TP_reco_v2_complet.ipynb` (export `TP_reco_v2_execute.html`) ; `TP_reco_v2.ipynb` est le notebook d'appui des compléments (autres réglages, seuil de similarité, règle à un écart-type). Rendre l'une ou l'autre version, avec son rapport et son export HTML ; le code des modules est le même.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace TP_reco.ipynb   # tout ré-exécuter (~15 min, grille en cache dans resultats/)
uv run jupyter nbconvert --to html --embed-images --output TP_reco_execute.html TP_reco.ipynb   # export avec les sorties
```

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), qui installe aussi Python tout seul si besoin.

- **Windows** (PowerShell) :
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  puis fermer et rouvrir le terminal. Vérifier avec `uv --version`.
  Alternative si `winget` est disponible : `winget install --id=astral-sh.uv -e`.
- **Linux / macOS** :
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

```bash
git clone https://github.com/albanrouault/ESIEA_5A_INF5063_syst-de-reco.git
cd ESIEA_5A_INF5063_syst-de-reco
uv sync                            # crée .venv et installe les dépendances
uv run nbstripout --install        # à faire UNE FOIS par clone (voir ci-dessous)
```

### Données

Les trois fichiers utilisés par le TP sont **versionnés** dans `Ressources/dataset/` (37 Mo),
rien à télécharger :

```
Ressources/dataset/movies_metadata.csv
Ressources/dataset/ratings_small.csv
Ressources/dataset/links_small.csv
```

Le reste de [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
(`ratings.csv`, `credits.csv`, `keywords.csv`, `links.csv`, 860 Mo) n'est pas versionné : GitHub
refuse les fichiers de plus de 100 Mo. Ces fichiers sont ignorés par git, on peut les laisser dans
le dossier sans risque.

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
