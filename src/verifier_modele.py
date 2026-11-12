"""Rejoue les métriques du TP2 sur le même découpage."""
from math import isclose

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from xai_commun import CAS, contexte, enregistrer, provenance


def verifier(ctx):
    pred = ctx.m.predict(ctx.Xte)
    proba = ctx.m.predict_proba(ctx.Xte)[:, ctx.classe_fraude]
    source = ctx.client.get_run(ctx.version.run_id)
    f1 = f1_score(ctx.yte, pred)
    f1_source = source.data.metrics["f1"]
    if not isclose(f1, f1_source, rel_tol=0, abs_tol=0.00005):
        raise ValueError(f"F1 recalculé {f1:.6f} différent du run source {f1_source:.6f}")
    return enregistrer("verification.json", {
        **provenance(ctx), "classe": type(ctx.m).__name__,
        "n_estimators": ctx.m.n_estimators, "max_depth": ctx.m.max_depth,
        "lignes": ctx.lignes, "fraudes": ctx.fraudes, "test_lignes": len(ctx.Xte),
        "test_fraudes": int(ctx.yte.sum()), "f1": f1, "f1_source": f1_source,
        "precision": precision_score(ctx.yte, pred), "rappel": recall_score(ctx.yte, pred),
        "auc": roc_auc_score(ctx.yte, proba), "matrice": confusion_matrix(ctx.yte, pred).tolist(),
        "benali_proba": float(ctx.m.predict_proba(CAS)[0, ctx.classe_fraude]),
    })


if __name__ == "__main__":
    verifier(contexte())
