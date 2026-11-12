"""Exécute les sept activités sur une version et un fichier de données épinglés."""
import argparse
import platform
from datetime import date
from importlib.metadata import version

import shap

from interpretables import comparer
from journaliser import empreintes, journaliser
from publier_card import publier_card
from verifier_modele import verifier
from xai_commun import REGLAGES, REPORTS, contexte, enregistrer, provenance
from xai_global import expliquer_global
from xai_importance import importance
from xai_lime import expliquer_lime
from xai_local import expliquer_local


def executer(date_audit):
    ctx = contexte()
    (REPORTS / "audit_execution.json").unlink(missing_ok=True)
    verifier(ctx)
    importance(ctx)
    explainer = shap.TreeExplainer(ctx.m)
    expliquer_global(ctx, explainer)
    local = expliquer_local(ctx, explainer)
    expliquer_lime(ctx)
    comparer(ctx)
    noms = [
        "verification.json", "importance.json", "permutation.png",
        "shap_importance.json", "shap_global.png", "shap_montant.png", "shap_local.json",
        "lime_stabilite.json", "lime_benali.html", "lime_benali.png", "interpretables.json",
        *(f"shap_{nom}.png" for nom in local),
    ]
    enregistrer("audit_execution.json", {
        "date": date_audit, "provenance": provenance(ctx), "reglages": REGLAGES,
        "environnement": {
            "python": platform.python_version(),
            **{nom: version(nom) for nom in ("mlflow", "scikit-learn", "numpy", "pandas", "shap", "lime")},
        },
        "artefacts_sha256": empreintes(REPORTS / nom for nom in noms),
    })
    publier_card(journaliser(ctx), date_audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-audit", type=date.fromisoformat, default=date.today())
    executer(parser.parse_args().date_audit.isoformat())
