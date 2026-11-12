# TP07 — Explicabilité et auditabilité du modèle de fraude

Ismail Lahlou · AI · G6

Le projet charge `fraude@production` depuis le registre MLflow du TP2, épingle la version résolue pour l’audit, puis vérifie ses scores. Il compare MDI, permutation, TreeSHAP et LIME ainsi qu’une régression logistique et des arbres courts sur le même test.

## Exécution

Démarrer le serveur MLflow dans le dossier `tp2-mlflow`, avec son environnement et sa base d’origine.

```powershell
.venv\Scripts\python.exe -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Dans le dossier de ce dépôt, avec Python 3.12 et le serveur précédent en cours d’exécution :

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item ../tp2-mlflow/data/transactions.csv data/
.venv\Scripts\python.exe src/executer.py
```

Adapter le chemin de la copie si le TP2 se trouve ailleurs. `donnees.py` régénère le jeu synthétique du TP2 avec la même graine. Pour un serveur sur un autre port, définir `MLFLOW_TRACKING_URI`.

## Fichiers utiles

- `verification.json` compare les métriques recalculées à celles du run du TP2.
- `importance.json`, `permutation.png`, `shap_importance.json`, `shap_global.png` et `shap_montant.png` documentent les explications globales.
- `shap_local.json` contient les deux transactions, les contributions additives et la phrase destinée à Benali. Les fichiers `shap_benali.png` et `shap_test_*.png` montrent leur décomposition.
- `lime_benali.html`, `lime_benali.png` et `lime_stabilite.json` couvrent cinq graines pour chacun des voisinages de 500 et 2 000 observations.
- `interpretables.json` contient les métriques, les coefficients de la logistique et les règles de l’arbre de profondeur 3.
- `audit_execution.json` identifie la version, les réglages et les empreintes des artefacts.
- `mlflow_validation.json` et `run_xai.txt` identifient le run MLflow des explications. Le run et la description de la version auditée contiennent la [Model Card](MODEL_CARD.md).

Les sorties sont dans `reports/`. La base MLflow, les binaires du modèle et le CSV restent locaux. Le run d’audit et les artefacts MLflow sont consultables depuis le registre qui a exécuté le TP.
