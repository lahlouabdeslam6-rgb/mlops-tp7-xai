# Model Card — fraude v1

## Modèle
Forêt aléatoire de 500 arbres, profondeur 16.
Version résolue à partir de l’alias `production` lors de l’audit.
Run d’entraînement MLflow `0d05920f9b6f45e5b4f99f6f5b4267db`.

## Objectif et usage prévu
Démonstration de détection de transactions suspectes et aide à la revue humaine.

## Usages exclus
Blocage définitif d’un compte ou décision réelle sur des clients à partir de ce seul modèle pédagogique.

## Données
Transactions synthétiques du TP2, générées avec la graine 7.
5000 lignes, dont 568 fraudes (11.36%).
MD5 de `data/transactions.csv` : `cf9f6f86c141e7df32c61715c493483f`.

## Performances
Test de 1250 transactions, dont 134 fraudes.
F1 0.8669, précision 0.8837, rappel 0.8507, AUC 0.9872.
Matrice de confusion `[[1101, 15], [20, 114]]` (classes 0 puis 1).

## Explicabilité
La permutation et TreeSHAP placent `montant` en tête. TreeSHAP global porte sur 400 observations de test.
Cas fictif Benali : score 0.9240, reconstruit par la base et les contributions SHAP. LIME approxime localement le comportement de la forêt et ses poids varient selon l’échantillonnage.

## Modèle interprétable
Logistique avec log10(montant) : F1 0.8593, 113 fraudes détectées sur 134.
Forêt : 114 sur 134. Cet écart dépend du jeu de test.

## Limites et risques
Données synthétiques, erreurs de détection et calibration en conditions réelles non évaluée.
SHAP décrit la prédiction du modèle. Une contribution ne suffit pas à établir une relation causale.

## Surveillance et revue
Suivre précision, rappel, dérive des entrées et faux positifs sur des données représentatives.
Rejouer l’audit après réentraînement ou changement d’alias, avant toute promotion.

## Traçabilité
Expérience `tp7-explicabilite`, run des explications `62f1d871d7e849f8bd3cd7257a2c4993`.
Ismail Lahlou (AI, G6), audit du 2026-11-12. Prochaine revue au prochain changement de version.
