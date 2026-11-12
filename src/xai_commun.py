"""Contexte reproductible de l'audit du modèle servi."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
REPORTS = ROOT / "reports"
NOM, ALIAS = "fraude", "production"
EXPERIENCE = "tp7-explicabilite"
VARIABLES = ["montant", "anciennete", "freq_24h", "nuit"]
CAS = pd.DataFrame([[1800.0, 30, 4, 1]], columns=VARIABLES)
REGLAGES = {
    "shap_echantillon": 400, "shap_random_state": 0,
    "lime_num_samples": 2000, "lime_random_state": 0,
    "permutation_n_repeats": 10, "permutation_random_state": 0,
    "test_size": 0.25, "split_random_state": 42,
}


def configurer_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return MlflowClient()


@lru_cache(maxsize=1)
def contexte():
    client = configurer_mlflow()
    version = client.get_model_version_by_alias(NOM, ALIAS)
    # L'alias est résolu une seule fois pour épingler la version durant l'audit.
    modele = mlflow.sklearn.load_model(f"models:/{NOM}/{version.version}")
    donnees = DATA.read_bytes()
    from io import BytesIO
    df = pd.read_csv(BytesIO(donnees))
    X, y = df[VARIABLES], df["fraude"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=REGLAGES["test_size"], random_state=REGLAGES["split_random_state"]
    )
    return SimpleNamespace(
        client=client, version=version, m=modele, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
        donnees_md5=hashlib.md5(donnees).hexdigest(), lignes=len(df), fraudes=int(y.sum()),
        classe_fraude=list(modele.classes_).index(1),
    )


def provenance(ctx):
    return {
        "modele": NOM, "alias": ALIAS, "modele_version": str(ctx.version.version),
        "modele_run_id": ctx.version.run_id, "donnees_md5": ctx.donnees_md5,
    }


def enregistrer(nom, contenu):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / nom).write_text(json.dumps(contenu, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contenu, ensure_ascii=False, indent=2))
    return contenu
