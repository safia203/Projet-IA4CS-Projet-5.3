Description du Projet :
Ce projet explore la sécurité des systèmes de détection de logiciels malveillants basés sur l'Intelligence Artificielle. Nous avons mis en œuvre une attaque adversariale (Bruit Gaussien) pour tester la robustesse d'un classifieur Random Forest entraîné sur le dataset EMBER.

Organisation du Dépôt: 
Le répertoire est structuré de façon à séparer les ressources techniques de la documentation :

 Code IA4CS/ : Contient les scripts de développement :

extract_subset.py : Script utilisé pour extraire un sous-ensemble (subset) du dataset EMBER original afin d'optimiser le temps de calcul.

train_randomforest.py : Script pour l'apprentissage du modèle sur les données extraites.

attaque.py : Algorithme de génération des exemples adversariaux par ajout de bruit.

model_rf.pkl, X_test.csv, y_test.csv : Modèle sauvegardé et données de test.

 Rapport/ : Documentation du projet :

projet 5.3 (1).pdf : Analyse détaillée des méthodes et des résultats.

 ember_subset.csv : Le dataset réduit généré par le script d'extraction .
