# visualisation.py

import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap

# 1. Distribution de la variable cible
def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="SHOT_MADE_FLAG", data=df, hue="SHOT_MADE_FLAG", ax=ax)
    ax.set_title("Distribution de la variable cible (0 = raté, 1 = réussi)")
    ax.set_xlabel("Shot Made Flag")
    ax.set_ylabel("Nombre de tirs")
    return fig

# 2. Répartition des tirs par joueur et type de tir
def plot_shot_type_by_player(df, player_order):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x="PLAYER_NAME",
        hue="SHOT_TYPE",
        hue_order=["2PT Field Goal", "3PT Field Goal"],
        order=player_order,
        ax=ax
    )
    ax.set_title("Répartition des tirs par joueur et type de tir")
    ax.set_xlabel("Joueur")
    ax.set_ylabel("Nombre de tirs")
    plt.setp(ax.get_xticklabels(), rotation=45)
    return fig

# 3. Heatmap : type de tir vs joueur
def plot_action_type_heatmap(df, player_order):
    df_crosstab = pd.crosstab(df["ACTION_TYPE_GROUPED"], df["PLAYER_NAME"])
    df_crosstab = df_crosstab[player_order]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Heatmap du type de tir par joueur")
    return fig

# 4. Fonction de dessin du terrain NBA (utilisée plus bas)
import matplotlib.patches as patches
from matplotlib.patches import Arc


def draw_nba_court(axis=None):
    if axis is None:
        fig = plt.figure(figsize=(9, 9))
        axis = fig.add_subplot(111, aspect='auto')
    else:
        fig = None

    axis.plot([-250, 250], [-47.5, -47.5], 'k-')     
    axis.plot([-250, -250], [-47.5, 422.5], 'k-')    
    axis.plot([250, 250], [-47.5, 422.5], 'k-')      
    axis.plot([-250, 250], [422.5, 422.5], 'k-')     

    axis.plot([-30, 30], [-10, -10], 'k-', lw=2)     

    axis.plot([-80, -80], [-47.5, 142.5], 'k-')
    axis.plot([80, 80], [-47.5, 142.5], 'k-')
    axis.plot([-60, -60], [-47.5, 142.5], 'k-')
    axis.plot([60, 60], [-47.5, 142.5], 'k-')
    axis.plot([-80, 80], [142.5, 142.5], 'k-')

    hoop = Arc((0, 0), 15, 15, theta1=0, theta2=360, lw=1.5, color='black')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, lw=1.5, color='black')
    axis.add_patch(hoop)
    axis.add_patch(restricted)

    axis.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, lw=1.5, color='black'))
    axis.add_patch(Arc((0, 142.5), 120, 120, theta1=180, theta2=360, lw=1.5, linestyle='--', color='black'))

    axis.plot([-220, -220], [-47.5, 92.5], 'k-')
    axis.plot([220, 220], [-47.5, 92.5], 'k-')
    axis.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, lw=1.5, color='black'))

    axis.add_patch(Arc((0, 422.5), 122, 122, theta1=180, theta2=360, lw=1.5, color='black'))

    axis.set_xlim(-250, 250)
    axis.set_ylim(-47.5, 470)
    axis.set_aspect(1)
    axis.axis('off')

    return fig, axis


# 5. Heatmap des zones de tir par joueur (adaptée pour joueurs sélectionnés)
def plot_shot_zone_area(df, selected_players=None):
    # Définition des couloirs (x_min, x_max)
    couloirs = {
        'Left Side(L)': [-250, -150],
        'Left Side Center(LC)': [-150, -50],
        'Center(C)': [-50, 50],
        'Right Side Center(RC)': [50, 150],
        'Right Side(R)': [150, 250],
        'Back Court(BC)': [-250, 250]  # zone en bas
    }

    y_min, y_max = 0, 422.5
    y_backcourt_min, y_backcourt_max = 422.5, 470

    # Si aucun joueur sélectionné, prendre tous
    if selected_players is None:
        selected_players = df['PLAYER_NAME'].unique()

    fig, axes = plt.subplots(
        nrows=(len(selected_players)+2)//3, ncols=3,
        figsize=(15, 5*((len(selected_players)+2)//3))
    )
    axes = axes.flatten()

    for i, joueur in enumerate(selected_players):
        ax = axes[i]
        draw_nba_court(ax)
        df_j = df[df['PLAYER_NAME'] == joueur]

        for zone, (x_min, x_max) in couloirs.items():
            y0, y1 = (y_backcourt_min, y_backcourt_max) if zone=='Back Court(BC)' else (y_min, y_max)
            count = df_j[df_j['SHOT_ZONE_AREA']==zone].shape[0]
            if count > 0:
                color_intensity = min(1, count / df_j['SHOT_ZONE_AREA'].value_counts().max())
                rect_patch = patches.Rectangle(
                    (x_min, y0), x_max - x_min, y1 - y0,
                    color=plt.cm.Blues(color_intensity),
                    alpha=0.7
                )
                ax.add_patch(rect_patch)
                ax.text((x_min+x_max)/2, (y0+y1)/2, str(count),
                        ha='center', va='center', fontsize=10, color='red')

        ax.set_title(joueur, fontsize=12)

    # Supprimer axes vides
    for k in range(len(selected_players), len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    return fig


# Palette bleu clair → bleu foncé
heat_palette_blue = LinearSegmentedColormap.from_list(
    "heat_blue",
    ["#E0F3FF", "#66B2FF", "#1A75FF", "#0033A0"]
)

def plot_all_players_heatmap_blue(df, selected_players=None):
    if selected_players is None:
        selected_players = df['PLAYER_NAME'].unique()

    n_players = len(selected_players)
    n_cols = 3
    n_rows = math.ceil(n_players / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, player in enumerate(selected_players):
        ax = axes[i]
        draw_nba_court(ax)

        sub = df[df['PLAYER_NAME'] == player]

        sns.kdeplot(
            x=sub['LOC_X'],
            y=sub['LOC_Y'],
            fill=True,
            cmap=heat_palette_blue,
            bw_adjust=0.8,
            alpha=0.7,
            levels=20,
            thresh=0.05,
            gridsize=100,
            ax=ax
        )

        ax.set_title(player, fontsize=10, pad=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

# 6. Scatter de tous les joueurs (adapté pour joueurs sélectionnés)
def plot_all_players_scatter(df, selected_players=None):
    if selected_players is None:
        selected_players = df['PLAYER_NAME'].unique()

    n_players = len(selected_players)
    n_cols = 3
    n_rows = math.ceil(n_players / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()

    for i, player in enumerate(selected_players):
        ax = axes[i]
        draw_nba_court(ax)

        sub = df[df['PLAYER_NAME'] == player]
        colors = sub['SHOT_MADE_FLAG'].map({1: 'blue', 0: 'red'})  # bleu = réussi, rouge = raté

        ax.scatter(sub['LOC_X'], sub['LOC_Y'], c=colors, alpha=0.6, s=15)
        ax.set_title(player, fontsize=10, pad=12)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

def plot_shot_zone_range(df, selected_players=None):
    # Tranches plus compactes (réduction de la hauteur)
    tranches = {
        'Less Than 8 ft.': (0, 80),
        '8-16 ft.': (80, 160),
        '16-24 ft.': (160, 240),
        '24+ ft.': (240, 350),
        'Back Court Shot': (350, 470)
    }

    # Filtrer les joueurs si demandé
    if selected_players:
        joueurs = [p for p in selected_players if p in df['PLAYER_NAME'].unique()]
    else:
        joueurs = df['PLAYER_NAME'].unique()

    # Créer la grille de subplots
    fig, axes = plt.subplots(
        nrows=(len(joueurs) + 2) // 3,
        ncols=3,
        figsize=(15, 5 * ((len(joueurs) + 2) // 3))
    )
    axes = axes.flatten()

    for i, joueur in enumerate(joueurs):
        ax = axes[i]
        draw_nba_court(ax)

        df_j = df[df['PLAYER_NAME'] == joueur]

        # Dessiner les tranches
        for zone, (y_min, y_max) in tranches.items():
            count = df_j[df_j['SHOT_ZONE_RANGE'] == zone].shape[0]
            if count > 0:
                # Intensité couleur relative au max du joueur
                color_intensity = min(1, count / df_j['SHOT_ZONE_RANGE'].value_counts().max())
                rect_patch = patches.Rectangle(
                    (-250, y_min), 500, y_max - y_min,
                    color=plt.cm.Blues(color_intensity),
                    alpha=0.7
                )
                ax.add_patch(rect_patch)
                # Afficher le nombre au centre
                ax.text(0, (y_min + y_max) / 2, str(count),
                        ha='center', va='center', fontsize=10, color='red')

        ax.set_title(joueur, fontsize=12)

    # Supprimer axes vides
    for k in range(len(joueurs), len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    return fig


# 7. Distance de tir vs résultat
def plot_shot_distance(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df, x="SHOT_DISTANCE", hue="SHOT_MADE_FLAG",
                 bins=30, kde=True, element="step", ax=ax)
    ax.set_title("Distribution de la distance de tir selon le résultat")
    return fig


# 8. Violin réussite vs distance
def plot_violin_distance(df, selected_players=None):

    if selected_players:
        df = df[df["PLAYER_NAME"].isin(selected_players)]

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="PLAYER_NAME",
        y="SHOT_DISTANCE",
        data=df,
        hue="PLAYER_NAME",       # couleur par joueur
        dodge=False,             # pas de séparation par hue
        palette="Blues",
        legend=False,
        inner="quartile"         # quartiles affichés
    )
    plt.xticks(rotation=45)
    plt.title("Distribution des distances de tir par joueur")
    plt.xlabel("Joueur")
    plt.ylabel("Distance du tir (pieds)")
    plt.tight_layout()
    
    fig = plt.gcf()  # récupérer la figure matplotlib pour Streamlit
    return fig


def plot_last_5d_pct(df, selected_players=None):
    """
    Trace l'évolution du pourcentage de réussite sur les 5 derniers jours par joueur.
    """
    # S'assurer que GAME_DATE est en datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    # Calcul du jour dans la saison (1er octobre = jour 0)
    def day_in_saison(date):
        year = date.year
        season_start = pd.Timestamp(year=year, month=10, day=1)
        if date < season_start:
            season_start = pd.Timestamp(year=year - 1, month=10, day=1)
        return (date - season_start).days

    df['day_in_season'] = df['GAME_DATE'].apply(day_in_saison)

    # Filtrer les joueurs si demandé
    if selected_players is None:
        selected_players = df['PLAYER_NAME'].unique()

    figs = []  # pour stocker toutes les figures

    for joueur in selected_players:
        df_j = df[df['PLAYER_NAME'] == joueur].copy()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(
            data=df_j,
            x='day_in_season',
            y='LAST_5D_PCT',
            hue='SEASON',
            linewidth=2,
            alpha=0.8,
            errorbar=None,
            ax=ax
        )
        ax.set_title(f"Pourcentage de réussite sur les 5 derniers jours - {joueur}")
        ax.set_xlabel('Jour dans la saison (1er octobre = jour 0)')
        ax.set_ylabel('Pourcentage sur 5 derniers jours')
        ax.set_ylim(0, 1)
        ax.legend(title='Saison')
        plt.tight_layout()

        figs.append(fig)
    
    return figs  # renvoie la liste des figures

