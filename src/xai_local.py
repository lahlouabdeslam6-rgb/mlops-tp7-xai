"""Décompose le score de deux transactions fictives ou synthétiques."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from xai_commun import CAS, REPORTS, contexte, enregistrer


def expliquer_local(ctx, explainer):
    proba_test = ctx.m.predict_proba(ctx.Xte)[:, ctx.classe_fraude]
    index = int(ctx.Xte.index[proba_test.argmax()])
    resultats = {}
    for nom, transaction in [("benali", CAS), (f"test_{index}", ctx.Xte.loc[[index]])]:
        exp = explainer(transaction)[0, :, ctx.classe_fraude]
        base = float(exp.base_values)
        somme = base + float(exp.values.sum())
        proba = float(ctx.m.predict_proba(transaction)[0, ctx.classe_fraude])
        np.testing.assert_allclose(somme, proba, atol=1e-6)
        resultats[nom] = {
            "transaction": transaction.iloc[0].to_dict(), "base": base, "score": proba,
            "base_plus_contributions": somme,
            "contributions": dict(zip(transaction.columns, exp.values.tolist())),
        }
        REPORTS.mkdir(exist_ok=True)
        shap.plots.waterfall(exp, show=False)
        plt.savefig(REPORTS / f"shap_{nom}.png", dpi=140, bbox_inches="tight")
        plt.close()
    resultats[f"test_{index}"]["classe_reelle"] = int(ctx.yte.loc[index])
    resultats["benali"]["phrase_client"] = (
        "Le montant de votre paiement et son horaire nocturne ont déclenché un contrôle. "
        "Nous pouvons vérifier la transaction avec vous."
    )
    return enregistrer("shap_local.json", resultats)


if __name__ == "__main__":
    ctx = contexte()
    expliquer_local(ctx, shap.TreeExplainer(ctx.m))
