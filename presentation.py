import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import math

def show_presentation():
    # Configuration de la page

    # Style CSS personnalisé
    st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: #1E64C8;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 28px;
            font-weight: bold;
            color: #2E86C1;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .equation-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0px;
        }
        .highlight {
            background-color: #e9f7fe;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #2E86C1;
        }
    </style>
    """, unsafe_allow_html=True)

    # Titre
    st.markdown('<p class="main-title">Flow-based Generative Models & Normalizing Flows</p>', unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    Les **Normalizing flows** sont une classe puissante de modèles génératifs en apprentissage automatique qui modélisent 
    explicitement une distribution de probabilité complexe en transformant une distribution simple via une série de 
    transformations inversibles.
    """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ## Concept fondamental
        
        Un flow-based generative model utilise la technique des normalizing flows pour transformer une distribution simple 
        (comme une gaussienne) en une distribution complexe qui peut représenter des données réelles. La clé de cette approche 
        réside dans le théorème de changement de variable pour les distributions de probabilité.
        
        Le principe fondamental : Transformer une variable aléatoire avec une distribution simple en une variable aléatoire 
        avec une distribution complexe via des transformations inversibles.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Code pour créer une visualisation simple de concept
        fig, ax = plt.subplots(figsize=(5, 4))
        # Dessiner une distribution gaussienne
        x = np.linspace(-3, 3, 1000)
        y_gauss = np.exp(-x**2/2) / np.sqrt(2*np.pi)
        ax.plot(x, y_gauss, 'b-', alpha=0.6, label='Distribution simple')
        
        # Dessiner une distribution complexe (multimodale)
        y_complex = 0.4*np.exp(-(x-1.5)**2/0.3) + 0.6*np.exp(-(x+1)**2/0.5)
        ax.plot(x, y_complex, 'r-', alpha=0.6, label='Distribution complexe')
        
        # Ajouter une flèche
        arrow = FancyArrowPatch((0.5, 0.6), (0, 0.8), 
                            connectionstyle="arc3,rad=.3", 
                            arrowstyle="Simple,head_width=10,head_length=10", 
                            color="green", lw=2)
        ax.add_patch(arrow)
        ax.text(0.2, 0.7, "Transformation", fontsize=10)
        
        ax.legend()
        ax.set_title('Concept des Normalizing Flows')
        ax.set_xlabel('x')
        ax.set_ylabel('Densité')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Méthode mathématique
    st.markdown('<p class="subtitle">Formulation mathématique</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="equation-box">
    Les normalizing flows définissent une séquence de transformations inversibles appliquées à une variable aléatoire.

    Soit $z$ une variable aléatoire avec une distribution simple $p_Z(z)$. Pour une séquence de $K$ transformations $f_1, f_2, ..., f_K$, on définit une séquence de variables aléatoires:

    $z_0 = z$
    $z_i = f_i(z_{i-1})$ pour $i = 1, 2, ..., K$

    La sortie finale $z_K$ modélise la distribution cible.

    La log-vraisemblance de $z_K$ est donnée par:

    $\log p_{Z_K}(z_K) = \log p_Z(z_0) - \sum_{i=1}^{K} \log|\det(J_{f_i}(z_{i-1}))|$

    où $J_{f_i}(z_{i-1})$ est la matrice jacobienne de $f_i$ évaluée en $z_{i-1}$.
    </div>
    """, unsafe_allow_html=True)

    # Avantages
    st.markdown('<p class="subtitle">Avantages des Flow-based Models</p>', unsafe_allow_html=True)

    avantages = [
        "Modélisation directe de la vraisemblance (likelihood)",
        "Possibilité de calculer et minimiser la log-vraisemblance négative comme fonction de perte",
        "Génération facile de nouveaux échantillons par échantillonnage de la distribution initiale",
        "Inversion exacte (contrairement aux VAE et GAN)",
        "Représentation explicite de la fonction de vraisemblance (contrairement aux VAE et GAN)"
    ]

    for i, avantage in enumerate(avantages):
        st.markdown(f"**{i+1}.** {avantage}")

    # Méthode d'entraînement
    st.markdown('<p class="subtitle">Méthode d\'entraînement</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        L'objectif d'entraînement est de minimiser la divergence de Kullback-Leibler (KL) entre la vraisemblance du modèle et la distribution cible. Cela revient à maximiser la vraisemblance du modèle sur les échantillons observés.
        
        <div class="equation-box">
        La divergence KL est définie comme:
        
        $D_{KL}(p_{data} || p_{model}) = \mathbb{E}_{x \sim p_{data}}[\log p_{data}(x) - \log p_{model}(x)]$
        
        Comme le premier terme est constant par rapport aux paramètres du modèle, on peut simplifier l'objectif à:
        
        $\mathcal{L} = \mathbb{E}_{x \sim p_{data}}[-\log p_{model}(x)]$
        
        En pratique, cette espérance est approximée par la moyenne sur un ensemble de données.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Pseudo-code pour l'entraînement
        st.code("""
    # Pseudo-code pour l'entraînement
    # INPUT: dataset D, normalizing flow model M
    # SOLVE: en minimisant la log-vraisemblance négative
    #        par descente de gradient

    def train_normalizing_flow(D, M, optimizer, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in DataLoader(D):
                # Calculer la log-vraisemblance négative
                z, ldj = M(batch)  # z: variable latente, ldj: log det jacobien
                prior_ll = log_prob_prior(z)  # log prob. distribution simple
                loss = -(prior_ll + ldj).mean()
                
                # Mise à jour des paramètres
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss}")
        
        return M
        """, language="python")

    # Section Limitations
    st.markdown('<p class="subtitle">Limitations et inconvénients</p>', unsafe_allow_html=True)

    limitations = [
        {
            "titre": "Espace latent de même dimension",
            "description": "Contrairement aux VAE, l'espace latent des normalizing flows n'est pas de dimension inférieure, " +
                        "ce qui ne permet pas de compression des données par défaut et nécessite plus de calculs."
        },
        {
            "titre": "Difficulté avec les échantillons hors distribution",
            "description": "Les modèles flow-based sont connus pour échouer dans l'estimation de la vraisemblance " +
                        "d'échantillons hors distribution (échantillons qui ne proviennent pas de la même distribution que l'ensemble d'entraînement)."
        },
        {
            "titre": "Problèmes d'inversibilité numérique",
            "description": "Bien que théoriquement inversibles, en pratique, l'inversibilité peut être violée à cause " +
                        "d'imprécisions numériques, ce qui peut mener à des explosions dans la fonction inverse."
        },
        {
            "titre": "Complexité de calcul",
            "description": "Le calcul des déterminants des matrices jacobiennes peut être coûteux, limitant l'application " +
                        "à des problèmes de très grande dimension sans conception attentive de l'architecture."
        }
    ]

    col1, col2 = st.columns(2)

    for i, limitation in enumerate(limitations):
        if i % 2 == 0:
            with col1:
                st.markdown(f"""
                **{limitation['titre']}**  
                {limitation['description']}
                """)
        else:
            with col2:
                st.markdown(f"""
                **{limitation['titre']}**  
                {limitation['description']}
                """)

    # Applications
    st.markdown('<p class="subtitle">Applications des Normalizing Flows</p>', unsafe_allow_html=True)

    applications = {
        "Génération audio": "Création de sons, parole et musique réalistes",
        "Génération d'images": "Création d'images photoréalistes et manipulation d'images",
        "Génération de graphes moléculaires": "Conception de nouvelles molécules pour la découverte de médicaments",
        "Modélisation de nuages de points": "Génération et analyse de données 3D",
        "Génération vidéo": "Création de séquences vidéo temporellement cohérentes",
        "Compression d'images avec perte": "Techniques de compression basées sur les flows",
        "Détection d'anomalies": "Identification d'échantillons anormaux ou outliers"
    }

    # Affichage en grille
    cols = st.columns(3)
    for i, (app, desc) in enumerate(applications.items()):
        col_index = i % 3
        with cols[col_index]:
            st.markdown(f"""
            **{app}**  
            {desc}
            """)
            st.write("---")

    # Conclusion et référence
    st.markdown('<p class="subtitle">Conclusion</p>', unsafe_allow_html=True)

    st.markdown("""
    Les normalizing flows représentent une approche puissante et mathématiquement élégante pour la modélisation 
    générative. Bien qu'ils présentent certaines limitations, leur capacité à modéliser explicitement des distributions 
    complexes et à générer de nouveaux échantillons en fait un outil précieux dans de nombreux domaines d'application.

    La recherche continue d'améliorer ces modèles, en particulier pour traiter les limitations mentionnées précédemment, 
    ouvrant la voie à des applications toujours plus avancées en intelligence artificielle.
    """)

    # Références
    st.markdown('<p class="subtitle">Références</p>', unsafe_allow_html=True)

    st.markdown("""
    1. Tabak, E. G., & Vanden-Eijnden, E. (2010). Density estimation by dual ascent of the log-likelihood.
    2. Tabak, E. G., & Turner, C. V. (2012). A family of nonparametric density estimation algorithms.
    3. Papamakarios, G., et al. (2021). Normalizing flows for probabilistic modeling and inference.
    4. Dinh, L., et al. (2014). NICE: Non-linear Independent Components Estimation.
    5. Dinh, L., et al. (2016). Density estimation using Real NVP.
    6. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions.
    7. Kobyzev, I., et al. (2021). Normalizing Flows: An Introduction and Review of Current Methods.
    """)

    # Ajouter un footer
    st.markdown("---")
    st.markdown("Développé pour une présentation éducative sur les Normalizing Flows | Basé sur Wikipedia")