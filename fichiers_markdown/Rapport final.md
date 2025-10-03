# 📑 Rapport final



### ✅ Conclusion tirées

L’analyse de 4 saisons de tirs NBA extraites via nba_api a montré que les performances au tir sont fortement contextuelles : la distance au panier (SHOT_DISTANCE), les cumuls récents (LAST_5_PCT / LAST_5D_PCT), l’angle du tir (ANGLE_TO_HOOP) et le contexte de score (SCOREMARGIN) sont des prédicteurs robustes. Les modèles classiques (RandomForest, XGBoost) obtiennent des performances stables autour de 0.62–0.66 d’accuracy et 0.64–0.66 d’AUC ROC selon les itérations documentées. Le deep learning (MLP wide&deep, TabNet) n’a pas dépassé significativement ces scores sur le jeu disponible, indiquant que la richesse des features et la variabilité individuelle des joueurs pèsent plus que la complexité du modèle.

---

### ⚠️ Difficultés rencontrées lors du projet

Le principal verrou scientifique a été la difficulté à obtenir de données brut concernant le tracking des joueurs en temps reel sur le terrain, ce qui aurait permis de connaitre la proximité du défenseur ou d'un coequipier. Le basket_ball étant un sport d'équipe et d'aversité le poids de l'aide apportés par les coéquipiers n'a pas pu être capturé de manière significative, de même l'impact des defenseur n'a pu être obtenu que par approximation, en reconstruisant des features à partir de données déjà aggrégées à l'echelle d'un match et non action par action comme pour le dataset de base contenant les shoots. Par defaut nous n'avons pas pu obenir au moins une variable shoot contesté/shoot libre. Ces données sont proprietés des équipes de NBA et ne sont pas accessible librement. La reconstruction de ces variables par approximation à considerablement dilué l'impact de ces features. 

#### ⏳ Prévisionnel : tâches qui ont pris plus de temps que prévu

- Recuperation des datas pertinentes à partirs d'un multitude de dataframe servi par les endpoints de NBA_API. La plupart des données étant déjà aggregé par match ou par saison, ce qui complique et limite leur utilisation dans le cadre de notre sujet qui se situe à l'echelle action par action.
- Feature engenneering, reconstruction de variable utilisable pour notre modele à partir de ces données aggrégées.
- Conception et vérification des cumuls causaux (shift/ordering) : plusieurs itérations pour garantir absence de fuite temporelle.
- Tuning de modèles (GridSearch / Randomized / Optuna) sur validation temporelle : longue consommation CPU et itérations répétées.
- Intégration et extraction de features avancées (ANGLE_TO_HOOP, défensives agrégées) : complexité mathématique et vérifications géométriques.

#### 📊 Jeux de données : acquisition, volumétrie, traitement, agrégation

- Acquisition : récupération via nba_api sur 4 saisons — reproductible mais lente (API rate limits et nettoyage nécessaire).
- Volumétrie : ~37k tirs documentés dans le premier rapport, plusieurs centaines de milliers d’événements selon granularité PlayByPlay ; taille suffisante pour ML classique mais limitée pour DL profond.
- Traitement : nettoyage, imputation (ex SCOREMARGIN), calcul de cumuls/rolling, agrégation par joueur/zone, variables temporelles agregees pour caturer la fatigue, reconstructions de variables pour capturer l'impact de la défense à partir de données par match.

#### 🧑‍💻 Compétences techniques / théoriques

- Acquisition rapide de compétences pratiques en XGBoost, TabNet et Keras; montage d’un pipeline ColumnTransformer/OneHot/Scaler.
- Difficultés initiales : implémentation correcte des validations temporelles (TimeSeriesSplit vs split par date), gestion des features hiérarchiques (effet joueur), et interprétabilité (mise en place SHAP).

#### 🎯Pertinence : de l’approche, du modèle, des données

- Approche : pertinente (features contextuelles + rolling), mais perfectible : régularisation bayésienne souhaitable pour cumuls sur petits effectifs et modèles hiérarchiques (mixed-effects) pour capturer l’effet joueur.
- Modèles : XGBoost offre le meilleur compromis performance/temps. DL intéressant mais limité par volume et signal disponible.
- Données : nécessité d’enrichir par variables défensives par possession, tracking player-level (if available), et contexte d’équipe adversaire.

#### 🖥️ IT : puissance de stockage, puissance computationnelle

- Entraînements et Grid/Random Search lourds en CPU; Optuna utile mais demande ressources.
- Deep learning (Keras/TabNet) demande GPU pour accélérer; absence de GPU rend l’itération plus lente.
- Stockage: jeux csv/npz et objets picklés suffisamment petits, mais snapshots d’historique et versions de preprocessors à sauvegarder pour reproductibilité.

#### 📌 Autres

- Visualisations interactives (Streamlit) faciles à prototyper mais la configuration des exports (pickle/keras save) nécessite des conventions robustes pour déploiement.

---

### 📈 Bilan

#### 🌟 Contribution principale

- Conception et implémentation du pipeline end-to-end : ingestion via nba_api, nettoyage, feature engineering (cumuls par zone/type, rolling windows, ANGLE_TO_HOOP, DAYS_SINCE_LAST_GAME), pipelines sklearn (ColumnTransformer), évaluation et prototypes modèles (RandomForest, XGBoost, Keras wide&deep, TabNet).
- Mise en place d’une application Streamlit/visualisation (notebook / app scaffolding) pour inspection des métriques, confusion matrices et importance de features.

#### 🔄 Evolution des modèles

- Oui. Itérations documentées :
  - Passage d’un RandomForest brut (baseline) à GridSearchCV optimisé (meilleure profondeur et n_estimators) -> gain accuracy ~0.03.
  - Introduction XGBoost optimisé (learning_rate, max_depth, subsample) -> meilleure CV score.
  - Ajout d’un modèle wide&deep Keras et TabNet pour tester capacités non linéaires profondes ; gains marginaux nuls au regard du travail d’ingénierie des features.
  - Ajout de transformations standardisées et d’un encodage dense pour compatibilité Keras.

#### 📊 Résultats obtenus

- RandomForest (non optimisé) : accuracy ≈ 0.60, AUC ≈ 0.64 (ex Rapport_2).
- RandomForest (GridSearch) : accuracy ≈ 0.63, amélioration notable sur recall pour « Miss ».
- XGBoost (GridSearch) : accuracy ≈ 0.65, best CV score ≈ 0.657 ; meilleure séparation et AUC plus élevée.
- Keras / TabNet : accuracy ≈ 0.62–0.63, AUC ≈ 0.65 ; stabilité mais pas de dépassement significatif.
  Benchmark attendu (baseline raisonnable) : prédiction par zone/type (taux historique) — nos modèles améliorent ce baseline en captant interactions contextuelles, mais la marge d’amélioration reste modeste.

#### 🎯 Atteintes des objectifs

- Explorer et visualiser les données : Atteint — heatmaps, distributions, confusion matrices, feature importances et SHAP.
- Construire variables cumulées / temporelles : Atteint — LAST_5*, LAST_5D*, TOTAL_CUM_*, ANGLE_TO_HOOP, DAYS_SINCE_LAST_GAME.
- Identifier patterns de jeu : Partiellement atteint — zones et types préférés identifiés globalement; personnalisation par joueur limitée par variance et petits effectifs.
- Préparer dataset final pour classification supervisée : Atteint — pipeline preprocess prêt pour entraînement reproductible.
- Développer modèle prédictif : Atteint — plusieurs modèles entraînés et comparés ; performances solides mais perfectibles.

#### 🏀 Process métier où le modèle peut s’inscrire

- Aide à la décision tactique : évaluer probabilité de réussite par zone/joueur en situation de jeu (scouting, préparation d’adversaire).
- Analyse de performance joueur : détection de hot/cold streaks via LAST_5* et alertes.
- Valorisation/choix de recrutement : probabilité ajustée par contexte pouvant servir en complément d’évaluation de scouting.

---

### 🚀 Suite du projet

#### 📌 Pistes d’amélioration

1. Enrichir les features défensives par possession (opponent zone defense metrics, defender distance from shooter) si tracking data disponible.
2. Passer XGBoost/LightGBM avec feature interactions automatiques et tuning bayésien systématique (Optuna) sur CV temporel.
3. Modelisation hierarchique pour integrer les variables à differentes echelles (mixed effects) 
4. Calibration fine des probabilités (isotonic/platt) pour applicabilité opérationnelle.
5. Data augmentation temporelle : inclure sequences via RNN/transformer sur shots récents (si volume de séquences joueur suffisant).
6. Déploiement GPU pour accélérer l’itération deep learning et l’hyperparam tuning.

#### 🔬 Contribution scientifique

- Validation empirique que, dans le contexte des tirs NBA, l’ingénierie des features (cumuls causaux, angle) est plus critique que la complexité du modèle.
- Mise en évidence de la nécessité de méthodes hiérarchiques et de régularisation pour traiter la variance inter‑joueurs.
- Documentation reproductible du pipeline (preprocessing, validation temporelle) applicable à d’autres études d’événements sportifs ponctuels.

### 📅 Annexes

#### 📆 Diagramme de Gantt (texte résumé)

- Semaine 1–2 : collecte des données via nba_api, exploration initiale, définition des variables clés.
- Semaine 2–3 : implémentation du préprocessing, calcul des cumuls et features temporelles, vérifications causales, baselines (RandomForest,XGBOOST), visualisations et premières évaluations.
- Semaine 3–4 : tuning GridSearch/Randomized/XGBoost, validation temporelle.
- Semaine 4-5 : prototypes deep learning (Keras wide&deep, TabNet), évaluation et comparaison.
- Semaine 5-6 : intégration Streamlit, calibration et préparation du rapport final.

#### 📂 Description des fichiers de code fournis

- app.py (Streamlit) : interface pour  gerer les pages multiples.
- data_page.py: affichage du dataset
- vizualisation.py et vizualisation_page.py: construction des graphes de dataviz à partir de certaines variables du dataset
- models.py et models_page.py (version ML):
  - Définit les listes de colonnes par type (numerical_continuous, numerical_counts, categorical, boolean, target).
  - get_preprocessor(X) : construit ColumnTransformer (StandardScaler/OneHotEncoder/passthrough).
  - create_pipeline(model, X) : Pipeline(preprocessor, classifier).
  - train_and_eval(...) : entraîne pipeline, retourne classification_report, AUC et figure matrice de confusion.
  - grid_search_pipeline(...) : GridSearchCV avec TimeSeriesSplit, extraction des feature importances et figure top-20.
  - randomized_search_pipeline(...) et optuna_tune(...) : recherche d’hyperparamètres via RandomizedSearchCV et Optuna.
  - evaluate_model(...) : utilisation orientée Streamlit pour afficher rapports, matrice de confusion et importances (si fournies).
  - Points d’attention : gestion des get_feature_names_out() avec fallback, compatibilité avec différents estimateurs.
  - Gere les sauvegardes/chargements des modeles 

- deep_learning.py et deep_learning_page.py (partie Keras / TabNet)
  
  - preprocess_data(X_train, X_test, ...) : ColumnTransformer pour features continues, catégorielles (one‑hot dense) et booléennes ; renvoie arrays float32 et preprocessor.
  - build_wide_deep(input_dim) : architecture Keras wide & deep, batchnorm/dropout, compile avec Adam.
  - build_tabnet() : constructeur TabNetClassifier (paramètres proposés).
  - train_and_eval(...) : entraine Keras ou TabNet selon model_type, gère early stopping, retourne model, report, auc, figures et history.
  - eval_deep(...) : évalue modèle deep, construit matrice de confusion et trace l’historique si disponible; compatible Keras et TabNet.
  - prepare_save_object(...) : prépare dict pour sauvegarde (modèle, preprocessor, history) avec avertissement sur pickling.
  - Gere les sauvegardes/chargement des modeles

- prediction.py et prediction_page.py: petit jeu de prediction en live avec XGBOOST simple, reglages des 3 features plus importantes par l'utilisateur.

- requirements.txt : versions exactes de packages (scikit-learn, xgboost, tensorflow, pytorch/tabnet, shap, optuna, joblib).
- notebook_repro.ipynb : exécution pas-à-pas du pipeline sur un sous-ensemble échantillon.

---
