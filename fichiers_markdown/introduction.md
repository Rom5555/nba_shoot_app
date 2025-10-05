



### *4 saisons, des milliers de tirs, un pipeline intelligent*

---

### 🎯 Contexte du projet

Dans l’univers ultra-compétitif de la NBA, chaque tir compte. L’analyse fine des données de match permet aujourd’hui d’optimiser les performances individuelles, d’affiner les stratégies collectives et de guider les décisions tactiques. Ce projet s’inscrit dans cette dynamique en exploitant les données officielles de la NBA (via l’API `nba_api`) pour modéliser la probabilité de réussite d’un tir et élaborer des modeles de classification les plus performants possibles.

---

### 🧠 Objectifs

- Explorer les tirs de 8 des 20 meilleurs joueurs NBA du XXIe siecle encore en activité, sur 4 saisons (shotchartdetail, playbyplayV2, playercareerstats).
- Créer des variables contextuelles, temporelles et géométriques (ex : `ANGLE_TO_HOOP`, `DAYS_SINCE_LAST_GAME`, `LAST_5_PCT`).
- Construire un pipeline prédictif reproductible (RandomForest, XGBoost, Keras).
- Identifier les zones rentables, les types de tirs préférés et les dynamiques individuelles.

---

### 🔍 Approche technique

- **Feature engineering avancé** : cumuls causaux, rolling windows, imputation, conversion temporelle.
- **Modélisation supervisée** : RandomForest, XGBoost, Wide&Deep, TabNet.
- **Validation temporelle** : TimeSeriesSplit, tuning par GridSearch / Optuna.
- **Visualisation** : heatmaps, confusion matrices, SHAP, Streamlit interactif.



### 🚀 Applications métier

- **Scouting & préparation tactique** : prédiction par zone/joueur.
- **Analyse de performance** : détection de hot/cold streaks.
- **Recrutement** : valorisation contextuelle des joueurs.


