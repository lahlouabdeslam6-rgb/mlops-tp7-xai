"""Archive les explications et leur provenance dans le registre TP2."""
import hashlib
import json

import mlflow

from xai_commun import EXPERIENCE, REGLAGES, REPORTS, contexte, enregistrer, provenance


def empreintes(fichiers):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in fichiers}


def journaliser(ctx):
    manifeste = json.loads((REPORTS / "audit_execution.json").read_text(encoding="utf-8"))
    fichiers = [REPORTS / nom for nom in manifeste["artefacts_sha256"]]
    if manifeste["provenance"] != provenance(ctx) or empreintes(fichiers) != manifeste["artefacts_sha256"]:
        raise ValueError("Résultats incohérents avec l'audit. Relancer src/executer.py.")
    metriques = json.loads((REPORTS / "verification.json").read_text(encoding="utf-8"))
    mlflow.set_experiment(EXPERIENCE)
    with mlflow.start_run(run_name=f"xai-fraude-v{ctx.version.version}") as run:
        mlflow.set_tags({
            **provenance(ctx), "donnees": "data/transactions.csv",
            "methodes": "permutation, TreeSHAP, LIME",
        })
        mlflow.log_params(REGLAGES)
        mlflow.log_metrics({k: metriques[k] for k in ("f1", "precision", "rappel", "auc")})
        for fichier in [*fichiers, REPORTS / "audit_execution.json"]:
            mlflow.log_artifact(str(fichier), artifact_path="explications")
    ctx.client.set_model_version_tag(ctx.version.name, ctx.version.version, "explications_run", run.info.run_id)
    (REPORTS / "run_xai.txt").write_text(run.info.run_id + "\n", encoding="utf-8")
    return run.info.run_id


if __name__ == "__main__":
    print(journaliser(contexte()))
