"""Génère la fiche modèle à partir de l'audit et vérifie sa publication."""
import json
from datetime import date
from pathlib import PurePosixPath

from xai_commun import EXPERIENCE, REPORTS, ROOT, configurer_mlflow, enregistrer


def lire(nom):
    return json.loads((REPORTS / nom).read_text(encoding="utf-8"))


def publier_card(run_id, date_audit):
    client = configurer_mlflow()
    run = client.get_run(run_id)
    verification = lire("verification.json")
    manifeste = lire("audit_execution.json")
    provenance = manifeste["provenance"]
    if any(run.data.tags.get(k) != v for k, v in provenance.items()):
        raise ValueError("Le run des explications ne correspond pas à cet audit.")
    nom, version = provenance["modele"], provenance["modele_version"]
    modele = client.get_model_version(nom, version)
    if modele.run_id != provenance["modele_run_id"]:
        raise ValueError("Le run d'entraînement ne correspond pas à la version auditée.")
    globales = lire("shap_importance.json")["importance"]
    local = lire("shap_local.json")["benali"]
    lr = lire("interpretables.json")["modeles"]["logistique_log10"]
    test_fraudes = verification["test_fraudes"]
    texte = f"""# Model Card — {nom} v{version}

## Modèle
Forêt aléatoire de {verification['n_estimators']} arbres, profondeur {verification['max_depth']}.
Version résolue à partir de l’alias `production` lors de l’audit.
Run d’entraînement MLflow `{modele.run_id}`.

## Objectif et usage prévu
Démonstration de détection de transactions suspectes et aide à la revue humaine.

## Usages exclus
Blocage définitif d’un compte ou décision réelle sur des clients à partir de ce seul modèle pédagogique.

## Données
Transactions synthétiques du TP2, générées avec la graine 7.
{verification['lignes']} lignes, dont {verification['fraudes']} fraudes ({verification['fraudes']/verification['lignes']:.2%}).
MD5 de `data/transactions.csv` : `{provenance['donnees_md5']}`.

## Performances
Test de {verification['test_lignes']} transactions, dont {test_fraudes} fraudes.
F1 {verification['f1']:.4f}, précision {verification['precision']:.4f}, rappel {verification['rappel']:.4f}, AUC {verification['auc']:.4f}.
Matrice de confusion `{verification['matrice']}` (classes 0 puis 1).

## Explicabilité
La permutation et TreeSHAP placent `{next(iter(globales))}` en tête. TreeSHAP global porte sur 400 observations de test.
Cas fictif Benali : score {local['score']:.4f}, reconstruit par la base et les contributions SHAP. LIME approxime localement le comportement de la forêt et ses poids varient selon l’échantillonnage.

## Modèle interprétable
Logistique avec log10(montant) : F1 {lr['f1']:.4f}, {lr['fraudes_detectees']} fraudes détectées sur {test_fraudes}.
Forêt : {verification['matrice'][1][1]} sur {test_fraudes}. Cet écart dépend du jeu de test.

## Limites et risques
Données synthétiques, erreurs de détection et calibration en conditions réelles non évaluée.
SHAP décrit la prédiction du modèle. Une contribution ne suffit pas à établir une relation causale.

## Surveillance et revue
Suivre précision, rappel, dérive des entrées et faux positifs sur des données représentatives.
Rejouer l’audit après réentraînement ou changement d’alias, avant toute promotion.

## Traçabilité
Expérience `{EXPERIENCE}`, run des explications `{run_id}`.
Ismail Lahlou (AI, G6), audit du {date_audit}. Prochaine revue au prochain changement de version.
"""
    fichier = ROOT / "MODEL_CARD.md"
    fichier.write_text(texte, encoding="utf-8")
    client.log_artifact(run_id, str(fichier))
    client.update_model_version(nom, version, description=texte)
    client.set_model_version_tag(nom, version, "model_card", f"MODEL_CARD.md, run {run_id}")
    version_publiee = client.get_model_version(nom, version)
    assert version_publiee.description.strip() == texte.strip()
    assert version_publiee.tags["explications_run"] == run_id
    assert version_publiee.tags["model_card"] == f"MODEL_CARD.md, run {run_id}"
    artefacts = sorted(a.path for a in client.list_artifacts(run_id, "explications"))
    assert {PurePosixPath(a).name for a in artefacts} == {
        *manifeste["artefacts_sha256"], "audit_execution.json"
    }
    assert any(a.path == fichier.name for a in client.list_artifacts(run_id))
    enregistrer("mlflow_validation.json", {
        **provenance, "run_xai": run_id, "statut": client.get_run(run_id).info.status,
        "artefacts": artefacts, "model_card_artifact": fichier.name, "description_verifiee": True,
    })


if __name__ == "__main__":
    publier_card((REPORTS / "run_xai.txt").read_text(encoding="utf-8").strip(), date.today().isoformat())
