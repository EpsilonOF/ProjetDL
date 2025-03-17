import streamlit as st
import torch
import matplotlib.pyplot as plt
import os
from train_nice import (
    setup_model, target_distribution, train_model, execute_mnist_nice, 
    generate_mnist_samples, MNISTDistribution
)

def show_nice():
    st.title("NICE: Non-linear Independent Components Estimation")
    st.write("""
    NICE (Non-linear Independent Components Estimation) est un des premiers modèles de Normalizing Flow 
    proposé par Dinh et al. en 2014. Il introduit le concept de couplage additif pour permettre 
    des transformations inversibles avec un jacobien facilement calculable.
    """)
    
    st.header("Principe")
    st.write("""
    NICE transforme des données complexes en un espace latent avec une distribution simple 
    via une série de transformations inversibles. La particularité de NICE est d'utiliser 
    des transformations à jacobien triangulaire, permettant un calcul efficace du déterminant.
    """)
    
    st.header("Architecture")
    st.write("""
    Dans NICE, la transformation est définie comme suit:
    1. L'entrée x est divisée en deux parties (x₁, x₂)
    2. La première partie x₁ reste inchangée
    3. La seconde partie est transformée: x₂' = x₂ + m(x₁)
    4. Où m est une fonction arbitraire (généralement un réseau de neurones)
    5. La sortie de la couche est (x₁, x₂')
    """)
    
    st.latex(r'''
    \begin{align}
    y_1 &= x_1 \\
    y_2 &= x_2 + m(x_1)
    \end{align}
    ''')
    
    st.header("Propriétés mathématiques")
    st.write("""
    NICE présente plusieurs propriétés mathématiques importantes:
    
    1. **Inversibilité**: La transformation est facilement inversible:
    """)
    
    st.latex(r'''
    \begin{align}
    x_1 &= y_1 \\
    x_2 &= y_2 - m(y_1)
    \end{align}
    ''')
    
    st.write("""
    2. **Déterminant du Jacobien**: La matrice jacobienne de cette transformation est triangulaire,
       ce qui implique que son déterminant est simplement le produit des éléments diagonaux, qui sont tous égaux à 1.
    """)
    
    st.latex(r'''
    \det \left( \frac{\partial y}{\partial x} \right) = 1
    ''')
    
    st.write("""
    3. **Conservation du volume**: NICE est une transformation préservant le volume, 
       ce qui simplifie le calcul de la vraisemblance.
    """)
    
    st.header("Couches de scaling")
    st.write("""
    Après plusieurs couches de couplage additif, NICE utilise une transformation de mise à l'échelle
    (scaling) pour permettre une expression plus riche des distributions:
    """)
    
    st.latex(r'''
    y = \exp(s) \odot x
    ''')
    
    st.write("""
    où s est un vecteur de paramètres apprenables et ⊙ est un produit élément par élément.
    """)
    
    st.header("Forces et limitations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forces")
        st.write("""
        - Inversibilité exacte et efficace
        - Calcul simple du jacobien (déterminant = 1)
        - Efficacité computationnelle
        - Apprentissage stable
        - Premier modèle démontrant la viabilité des Normalizing Flows
        """)
    
    with col2:
        st.subheader("Limitations")
        st.write("""
        - Expressivité limitée par les transformations additives
        - Nécessite beaucoup de couches pour modéliser des distributions complexes
        - Pas d'adaptation du volume local (contrairement à RealNVP)
        - Performance limitée sur les données de haute dimension
        """)
    
    st.header("Impact historique")
    st.write("""
    NICE a posé les fondations théoriques et pratiques des Normalizing Flows modernes. 
    Son approche de couplage a inspiré de nombreuses améliorations, notamment RealNVP 
    et Glow, qui ont étendu ses principes pour créer des modèles plus expressifs.
    
    Ce modèle a ouvert la voie à une nouvelle classe de modèles génératifs avec une 
    vraisemblance exactement calculable, contrairement aux VAE et aux GAN.
    """)

    st.header("Tester le modèle NICE")
    st.write("""
    Expérimentez avec le modèle NICE en ajustant différents hyperparamètres et en choisissant
    différents jeux de données:
    """)

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Hyperparamètres")
        n_layers = st.slider("Nombre de couches de couplage", min_value=2, max_value=12, value=4, step=1)
        hidden_dim = st.slider("Nombre de neurones cachés", min_value=32, max_value=256, value=128, step=32)
        
        st.subheader("Données")
        dataset = st.selectbox(
            "Jeu de données",
            ["Distribution 2D", "Image personnalisée", "MNIST"]
        )
        
        if dataset=="Image personnalisée":
            uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        elif dataset=="Distribution 2D":
            distribution = st.selectbox(
                "Type de distribution 2D",
                ["TwoMoons", "Spirale", "Anneaux", "Damier"]
            )
        elif dataset=="MNIST":
            st.write("""
            Le modèle NICE sera appliqué sur le jeu de données MNIST pour générer des chiffres manuscrits.
            """)
            mnist_trained = os.path.exists("data/nice_mnist.pth")
            if mnist_trained:
                st.info("Un modèle pré-entraîné pour MNIST est disponible.")
            
            train_new = st.checkbox("Entraîner un nouveau modèle", value=not mnist_trained)
            
            if train_new:
                iter_mnist = st.slider("Itérations d'entraînement (MNIST)", 
                                      min_value=1000, max_value=15000, value=5000, step=1000)
        
        generate_button = st.button("Générer", key="nice_generate")

    with col2:
        if dataset != "MNIST":
            st.subheader("Paramètres d'entraînement")
            temperature = st.slider("Température", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            n_samples = st.slider("Nombre d'échantillons", min_value=1, max_value=16, value=4, step=1)
            max_iter = st.slider("Itérations d'entraînement", min_value=1000, max_value=5000, value=2000, step=500)
        else:
            st.subheader("Génération MNIST")
            n_samples_mnist = st.slider("Nombre d'échantillons", min_value=5, max_value=50, value=25, step=5)
            grid_size = st.slider("Taille de la grille", min_value=3, max_value=10, value=5, step=1)
            mnist_model_path = "data/nice_mnist.pth"

    # Section pour afficher les résultats
    if generate_button:
        # Cas MNIST
        if dataset == "MNIST":
            with st.spinner("Traitement du modèle NICE pour MNIST..."):
                # Setup directories
                os.makedirs("data", exist_ok=True)
                os.makedirs("images/nice", exist_ok=True)
                
                # Train or load the model
                if train_new:
                    st.info(f"Entraînement d'un nouveau modèle NICE sur MNIST avec {iter_mnist} itérations...")
                    model = execute_mnist_nice(train=True, model_path=mnist_model_path)
                else:
                    st.info("Chargement du modèle pré-entraîné...")
                    model = execute_mnist_nice(train=False, model_path=mnist_model_path)
                
                # Generate samples
                samples_path = "images/nice/mnist_samples.png"
                samples = generate_mnist_samples(model, n_samples=n_samples_mnist, 
                                               grid_size=grid_size, save_path=samples_path)
                
                # Show results
                st.success("Génération terminée!")
                st.image(samples_path, caption="Chiffres MNIST générés par le modèle NICE", width=500)
                
                st.write(f"""
                Modèle NICE avec {model.num_layers} couches de couplage et {model.hidden_dim} neurones cachés.
                Les images générées sont des échantillons synthétiques créés par le modèle, qui a appris
                la distribution des chiffres manuscrits de MNIST.
                """)
        
        # Cas 2D ou Image personnalisée
        else:
            with st.spinner("Génération en cours..."):
                image_path = None
                dataset_type = "2d"
                
                if dataset == "Image personnalisée" and uploaded_file is not None:
                    # Ensure directory exists
                    os.makedirs("images/nice", exist_ok=True)
                    
                    with open("images/nice/uploaded_image.png", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        image_path = "images/nice/uploaded_image.png"
                    
                    # Afficher l'image
                    st.image(uploaded_file, caption="Image téléchargée", width=200)
                    dataset_type = "image"
                
                # Configuration et entraînement du modèle
                model, device, target = setup_model(image_path, num_layers=n_layers, 
                                                   hidden_dim=hidden_dim, dataset_type=dataset_type)
                xx, yy, zz = target_distribution(target, device, dataset_type=dataset_type)
                
                loss_hist = []
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

                cols = st.columns(int((max_iter // (max_iter//10))/2)) 
                cols2 = cols
                for it in range(max_iter):
                    optimizer.zero_grad()
                    x = target.sample(512).to(device)
                    loss = -model.log_prob(x).mean()

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()

                    loss_hist.append(loss.item())

                    if (it + 1) % (max_iter//10) == 0:
                        model.eval()
                        with torch.no_grad():
                            log_prob = model.log_prob(zz.view(-1, 2)).to('cpu').view(*xx.shape)
                        prob = torch.exp(log_prob)
                        prob[torch.isnan(prob)] = 0

                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.pcolormesh(xx, yy, prob.numpy(), cmap='coolwarm')
                        ax.set_aspect('equal', 'box')
                        ax.axis('off')
                        ax.set_title(f"Itération {it+1}", fontsize=10)
                        if it//(max_iter//10)>=len(cols):
                            cols2[(it // (max_iter//10))%len(cols)].pyplot(fig)
                        else:
                            cols[it // (max_iter//10)].pyplot(fig)
                        plt.close(fig)  # Libération mémoire

                        model.train()

                # Affichage final de la courbe de perte
                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(loss_hist, label="Perte")
                ax_loss.set_xlabel("Itération")
                ax_loss.set_ylabel("Perte")
                ax_loss.set_title("Historique de la perte")
                ax_loss.legend()
                st.pyplot(fig_loss)

                # Générer des échantillons
                model.eval()
                with torch.no_grad():
                    samples = model.sample(n_samples * n_samples).view(n_samples, n_samples, 2)
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(samples[:,:,0].flatten(), samples[:,:,1].flatten(), s=5, alpha=0.5)
                ax.set_aspect('equal', 'box')
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                ax.set_title("Échantillons générés par NICE")
                st.pyplot(fig)

                st.write(f"Modèle NICE avec {n_layers} couches de couplage, {hidden_dim} neurones cachés, et {max_iter} itérations d'entraînement")
    
    st.header("Application à des données complexes")
    st.write("""
    NICE peut être appliqué à des problèmes plus complexes que les distributions 2D, comme la génération
    d'images. Pour les données de haute dimension comme MNIST (28×28 = 784 dimensions), NICE utilise la
    même architecture avec des couches de couplage additives, mais avec un réseau plus large et plus
    profond pour capturer les dépendances complexes.
    
    Bien que NICE fonctionne sur MNIST, sa performance est limitée par sa nature transformative additive.
    Les modèles plus récents comme RealNVP et Glow offrent une meilleure expressivité pour modéliser
    des distributions plus complexes grâce à leurs transformations affines et convolutives.
    """)
    
    # Afficher des échantillons MNIST générés s'ils existent
    mnist_samples_path = "images/nice/mnist_samples.png"
    if os.path.exists(mnist_samples_path):
        st.image(mnist_samples_path, 
                caption="Exemples de chiffres MNIST générés par le modèle NICE", 
                width=400)
    
    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur NICE:
    
    - [Article original: "NICE: Non-linear Independent Components Estimation"](https://arxiv.org/abs/1410.8516)
    - [Tutorial sur les Normalizing Flows](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
    - [Notebook d'implémentation de NICE](https://github.com/bayesiains/nsf/blob/master/examples/flow_comparison.ipynb)
    """)