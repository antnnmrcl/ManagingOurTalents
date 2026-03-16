# MODEL CARD : HR Turnover Prediction

## 1. Objectif du modèle
• **Cas d’usage visé :** Prédiction du risque de départ (turnover) des employés pour aider les RH à cibler les actions de rétention.
• **Entrées :** Données tabulaires (âge, salaire, satisfaction, engagement, absences, département, performance, etc.).
• **Sorties :** Probabilité de départ (score de 0 à 1) et classification binaire (0 = Actif, 1 = Terminé).

## 2. Données d’entraînement
• **Dataset(s) utilisés :** [Human Resources Data Set (Kaggle)](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) par Dr. Rich Huebner.
• **Taille / diversité :**
  - **Nombre total d’échantillons :** ~312 employés.
  - **Répartition des classes :** Déséquilibrée (majorité d'employés actifs, minorité de départs). Équilibrée informatiquement lors de l'entraînement.
  - **Diversité :** Multiples départements, âges et données démographiques. Les variables sensibles (Genre, RaceDesc) ont été conservées pour vérifier l'équité du modèle.
• **Limites connues :** Dataset généré de manière synthétique (visée éducative). Biais géographique (grosse majorité d'employés dans le Massachusetts) et sectoriel (forte représentation du département Production). Petit volume de données empêchant la généralisation à grande échelle.

## 3. Performances
• **Métriques utilisées :** F1-Score (métrique principale face au déséquilibre), Accuracy, Precision, Recall, AUC-ROC.
• **Résultats :** Les résultats globaux démontrent que l'approche *Frugal AI* fonctionne : des modèles très simples (comme la Régression Logistique ou l'Arbre de Décision) atteignent des métriques de performance comparables aux modèles complexes lourds mis en compétition paramétrique (Random Forest / XGBoost).

## 4. Limites
• **Risques d’erreur connus :** En raison du faible jeu de données, les faux positifs (employé signalé à risque alors qu'il va rester) et faux négatifs sont possibles.
• **Situations non couvertes :** Le modèle prend une "photo" à l'instant T et manque d'analyse longitudinale fine (évolution dans le temps). Il ignore le contexte macroéconomique (crise, inflation, etc.).
• **Risques de biais :** Si le jeu de données originel favorise les promotions de certains genres ou origines, le modèle pourrait en hériter silencieusement. L'analyse de l'importance des variables (SHAP) permet de le surveiller.

## 5. Risques & mitigation
• **Risques de mauvaise utilisation :** Utiliser ces prédictions pour automatiser des licenciements ou discriminer préventivement à l'embauche (système punitif au lieu d'être un système de rétention bienveillant). Penser que corrélation (montrée par SHAP) vaut causalité directe.
• **Contrôles mis en place :**
  - **Explicabilité totale :** Chaque prédiction est passée dans SHAP et LIME pour justifier "pourquoi" l'alerte est levée.
  - **Vie privée (RGPD) :** Application stricte du hachage pseudo-anonyme (SHA-256) pour les noms, et conversion des dates de naissance en âge simple.
  - **Avertissements transverses :** Insistance sur le fait que l'IA ne fournit qu'une *aide à la décision* et ne remplace pas le jugement humain RH.

## 6. Énergie et frugalité
• **Poids du modèle :** Moins de 1 Mo.
• **Temps d’inférence :** Immédiat (< 0.1s) sur un CPU standard.
• **Énergie estimée (CodeCarbon) :** L'approche privilégie un modèle classique et sélectionné pour son ratio F1/Complexité. Le suivi par `CodeCarbon` intégré indique une émission en fraction absolue de gramme de CO₂ par entraînement.

## 7. Cyber
• **Sécurisation des entrées :** Pas de prompts en texte libre, minimisant les prompt injections. Les features passent par un scaler standardisé.
• **Secrets protégés :** Entièrement open-source, fonctionne 100% en local et "offline". La solution est dépourvue d'API keys exposées ou de base de données persistante non sécurisée.
