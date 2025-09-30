# models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score
import optuna
from xgboost import XGBClassifier

# =========================
# Colonnes par type
# =========================
numerical_continuous = [
    "SHOT_DISTANCE", "SCOREMARGIN", "ANGLE_TO_HOOP",
    "LAST_5_PCT", "LAST_5D_PCT", "DAYS_SINCE_LAST_GAME",
    "EVENT_PROGRESSION", "def_score_sum", "def_score_avg",
    "def_weight_total", "def_score_per_dist", "def_score_per_angle",
    'DIST_X_ANGLE','PERIOD_X_EVENTPROG','DIST_X_TIME','DIST_X_SCORE_X_CLUTCH',
    'ROLL5_X_TOTAL','DIST_X_ZONE_PCT','DIST_X_RANGE_PCT','ACTION_ZONE_PCT',
    'ACTION_DIST_PCT','PRESSURE_BEHIND_CLUTCH','ANGLE_DIST_ACTION'
]

numerical_counts = [
    "GAME_NUMBER_PLAYER",
    "STREAK_MADE", "STREAK_MISS", "CURRENT_STREAK"
]

categorical = [
    "PLAYER_ID", "TEAM_ID_DEF",
    "ACTION_TYPE_GROUPED", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "PERIOD_SHOT", "POSITION",
    "personIdDef_1", "personIdDef_2", "personIdDef_3"
]

boolean = [
    "PREV_MADE",
    "MADE_2_IN_A_ROW", "MADE_3_IN_A_ROW",
    "MISS_2_IN_A_ROW", "MISS_3_IN_A_ROW",
    "IS_CORNER_THREE", "IS_CLUTCH"
]

target = "SHOT_MADE_FLAG"

# =========================
# Préprocesseur & pipeline
# =========================
def get_preprocessor(X):
    num_cont = [c for c in numerical_continuous if c in X.columns]
    num_counts = [c for c in numerical_counts if c in X.columns]
    cat = [c for c in categorical if c in X.columns]
    bool_cols = [c for c in boolean if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_cont', StandardScaler(), num_cont),
            ('num_counts', StandardScaler(), num_counts),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
            ('bool', 'passthrough', bool_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def create_pipeline(model, X):
    preprocessor = get_preprocessor(X)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline

# =========================
# Fonctions d'entraînement et tuning
# =========================
def train_and_eval(model_or_pipeline, X_train, y_train, X_test, y_test):
    """Retourne pipeline et figures pour Streamlit"""
    pipeline = model_or_pipeline if isinstance(model_or_pipeline, Pipeline) else create_pipeline(model_or_pipeline, X_train)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    
    report = classification_report(y_test, y_pred, target_names=['Miss','Made'], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=['Miss','Made']).plot(cmap=plt.cm.Blues, ax=ax_cm)
    
    return pipeline, report, auc, fig_cm

def grid_search_pipeline(model, X_train, y_train, X_test, y_test, param_grid, cv=3, scoring='f1'):
    pipeline = create_pipeline(model, X_train)
    tscv = TimeSeriesSplit(n_splits=cv)
    
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_.named_steps['classifier']
    preprocessor = grid.best_estimator_.named_steps['preprocessor']
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        num_cols = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
        cat_cols = preprocessor.transformers_[2][2] if len(preprocessor.transformers_) > 2 else []
        bool_cols = preprocessor.transformers_[3][2] if len(preprocessor.transformers_) > 3 else []
        feature_names = list(num_cols) + list(cat_cols) + list(bool_cols)
    
    try:
        importances = best_model.feature_importances_
    except AttributeError:
        importances = np.abs(best_model.coef_[0])
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    fig_importance, ax_imp = plt.subplots(figsize=(10,8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20), ax=ax_imp)
    ax_imp.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    
    # Ne plus créer la confusion matrix ici
    return grid, fig_importance, importance_df


def randomized_search_pipeline(model, X_train, y_train, param_dist, n_iter=50, cv=3, scoring='f1'):
    pipeline = create_pipeline(model, X_train)
    tscv = TimeSeriesSplit(n_splits=cv)
    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
                                scoring=scoring, n_jobs=-1, verbose=2, random_state=42)
    search.fit(X_train, y_train)
    return search

def optuna_tune(model_class, param_space, X_train, y_train, n_trials=50, scoring="f1"):
    def objective(trial):
        params = param_space(trial)
        model = model_class(**params)
        pipeline = create_pipeline(model, X_train)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            if scoring == "f1":
                score = f1_score(y_val, preds, zero_division=0)
            elif scoring == "roc_auc":
                probas = pipeline.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, probas)
            scores.append(score)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_model = model_class(**study.best_params)
    pipeline_final = create_pipeline(best_model, X_train)
    pipeline_final.fit(X_train, y_train)
    
    return pipeline_final

def evaluate_model(model, X_test, y_test, model_name="Modèle", feature_importances=False, fig_importance=None):
    """
    Évalue un modèle déjà entraîné :
    - Classification report
    - AUC ROC
    - Confusion matrix
    - Optionnellement feature importances si feature_importances=True
    """
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import roc_auc_score

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Rapport de classification et AUC
    report = classification_report(y_test, y_pred, target_names=['Miss','Made'], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    st.subheader(f"📊 Évaluation du {model_name}")
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).T)
    st.write(f"AUC ROC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=['Miss','Made']).plot(cmap=plt.cm.Blues, ax=ax_cm)
    st.pyplot(fig_cm)

    # Feature importances
    if feature_importances and fig_importance is not None:
        st.write("### Top 20 Feature Importances")
        st.pyplot(fig_importance)


