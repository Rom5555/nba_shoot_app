# deep_learning.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import StepLR

# ------------------------
# Colonnes (exportées)
# ------------------------
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

# ------------------------
# Préprocessing
# ------------------------
def preprocess_data(X_train, X_test,
                    numerical_continuous=numerical_continuous,
                    numerical_counts=numerical_counts,
                    categorical=categorical,
                    boolean=boolean):
    """
    Retourne X_train_proc (np.float32), X_test_proc (np.float32), preprocessor.
    preprocessor est un sklearn ColumnTransformer prêt à transformer de nouveaux X.
    """
    num_features = list(numerical_continuous) + list(numerical_counts)
    cat_features = list(categorical)
    bool_features = list(boolean)

    num_transformer = StandardScaler()
    # OneHotEncoder sparse output False pour compatibilité et sérialisation
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('bool', 'passthrough', bool_features)
    ], remainder='drop')

    # Fit / transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_proc = np.asarray(X_train_proc, dtype='float32')
    X_test_proc = np.asarray(X_test_proc, dtype='float32')

    return X_train_proc, X_test_proc, preprocessor

# ------------------------
# Keras Wide & Deep
# ------------------------
def build_wide_deep(input_dim):
    input_layer = Input(shape=(input_dim,), name="input")

    x = Dense(256, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    wide = Dense(64, activation='relu')(input_layer)

    combined = Concatenate()([x, wide])
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ------------------------
# TabNet builder
# ------------------------
def build_tabnet():
    return TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=StepLR,
        seed=42
    )

# ------------------------
# Evaluation (renvoie les objets et affiche si show_on_streamlit=True)
# ------------------------
def eval_deep(model, X_test, y_test, history=None, plot_cm=True, plot_loss=True, show_on_streamlit=True):
    """
    model: objet Keras ou TabNet
    history: Keras History object OR dict-like (TabNet history) OR None
    Retour: (report_dict, auc_value, fig_cm, fig_loss_or_None)
    """
    y_test_arr = np.asarray(y_test).ravel()

    # Predictions probabilities
    if hasattr(model, "predict_proba"):
        # TabNet: predict_proba retourne array (n_samples, 2)
        y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
    else:
        # Keras model
        y_proba = model.predict(X_test).ravel()

    y_pred = (y_proba > 0.5).astype(int)

    report_dict = classification_report(y_test_arr, y_pred, target_names=['Miss', 'Made'], output_dict=True)
    auc_val = roc_auc_score(y_test_arr, y_proba)

    # Confusion matrix figure
    cm = confusion_matrix(y_test_arr, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Miss', 'Made']).plot(cmap=plt.cm.Blues, ax=ax_cm, colorbar=False)
    ax_cm.set_title("Confusion Matrix")

    # Loss / metrics figure
    fig_loss = None
    if plot_loss:
        hist_dict = None
        if history is not None:
            if hasattr(history, "history"):
                hist_dict = history.history
            elif isinstance(history, dict):
                hist_dict = history
        # If no explicit history provided, attempt to extract TabNet internal history
        if hist_dict is None and hasattr(model, "history") and isinstance(getattr(model, "history"), dict):
            hist_dict = getattr(model, "history")

        if hist_dict:
            fig_loss, ax_loss = plt.subplots()
            # Plot keys that might exist (Keras or TabNet keys)
            keys_to_plot = ['loss', 'val_loss', 'train_logloss', 'valid_logloss', 'train_loss', 'valid_loss', 'train_auc', 'valid_auc', 'auc']
            plotted = False
            for k in keys_to_plot:
                vals = hist_dict.get(k)
                if isinstance(vals, (list, tuple)) and len(vals) > 0:
                    ax_loss.plot(vals, label=k)
                    plotted = True
            if plotted:
                ax_loss.set_title("Évolution des métriques")
                ax_loss.legend()
            else:
                fig_loss = None

    if show_on_streamlit:
        st.subheader("Rapport de classification")
        try:
            df_report = pd.DataFrame(report_dict).T
            # si colonnes présentes (précision, rappel, f1-score) → les afficher
            if all(col in df_report.columns for col in ["precision","recall","f1-score"]):
                st.dataframe(df_report[["precision","recall","f1-score"]])
            else:
                st.dataframe(df_report)  # fallback si colonnes différentes
        except Exception:
            st.json(report_dict)
        st.write(f"**AUC ROC :** {auc_val:.4f}")
        st.pyplot(fig_cm)
        if fig_loss is not None:
            st.pyplot(fig_loss)
        else:
            st.warning("Pas d'historique disponible pour les courbes loss/métriques.")

    return report_dict, auc_val, fig_cm, fig_loss

# ------------------------
# Entraînement (retourne aussi history)
# ------------------------
def train_and_eval(X_train, y_train, X_test, y_test, model_type='keras', epochs=50, batch_size=1024):
    """
    Retourne:
      model, report_dict, auc_value, fig_cm, fig_loss_or_None, history_obj
    history_obj: Keras History ou dict (TabNet history) ou None
    """
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    if model_type == 'keras':
        model = build_wide_deep(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        report, auc_val, fig_cm, fig_loss = eval_deep(model, X_test, y_test, history=history, plot_cm=True, plot_loss=True, show_on_streamlit=False)
        return model, report, auc_val, fig_cm, fig_loss, history

    elif model_type == 'tabnet':
        model = build_tabnet()
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_name=['train','valid'],
            eval_metric=['logloss','auc'],
            max_epochs=epochs,
            patience=5,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # extraire l'historique TabNet (dictionnaire si disponible)
        tb_history = None
        if hasattr(model, "history") and isinstance(model.history, dict) and model.history:
            tb_history = model.history
        else:
            # tenter d'extraire depuis le callback container (compat)
            if hasattr(model, "_callback_container"):
                for cb in getattr(model._callback_container, "callbacks", []):
                    if hasattr(cb, "history") and isinstance(cb.history, dict) and cb.history:
                        tb_history = cb.history
                        break

        report, auc_val, fig_cm, fig_loss = eval_deep(model, X_test, y_test, history=tb_history, plot_cm=True, plot_loss=True, show_on_streamlit=False)
        return model, report, auc_val, fig_cm, fig_loss, tb_history

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# ------------------------
# Helpers pour sauvegarde / chargement d'objets
# ------------------------
def prepare_save_object(model, preprocessor, history):
    """
    Prépare le dict à sauvegarder. Attention: certains objets (p. ex. TabNet model) peuvent ne pas être picklables
    selon la version. Dans ce cas sauvegardez le modèle séparément.
    """
    return {"model": model, "preprocessor": preprocessor, "history": history}

# ------------------------
# End of module
# ------------------------
