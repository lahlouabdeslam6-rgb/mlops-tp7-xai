"""Explique globalement la probabilité de fraude avec TreeSHAP."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from xai_commun import REGLAGES, REPORTS, contexte, enregistrer


def expliquer_global(ctx, explainer):
    Xs = ctx.Xte.sample(REGLAGES["shap_echantillon"], random_state=REGLAGES["shap_random_state"])
    sv = explainer(Xs)[:, :, ctx.classe_fraude]
    reconstruite = sv.base_values + sv.values.sum(axis=1)
    np.testing.assert_allclose(reconstruite, ctx.m.predict_proba(Xs)[:, ctx.classe_fraude], atol=1e-6)
    scores = dict(sorted(zip(Xs.columns, np.abs(sv.values).mean(axis=0).tolist()),
                         key=lambda item: item[1], reverse=True))
    enregistrer("shap_importance.json", {"base": float(sv.base_values[0]), "importance": scores})
    shap.plots.beeswarm(sv, show=False)
    plt.savefig(REPORTS / "shap_global.png", dpi=140, bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots()
    nuage = ax.scatter(Xs["montant"], sv[:, "montant"].values, c=Xs["nuit"],
                       cmap="coolwarm", s=12)
    ax.set(xscale="log", xlabel="Montant (MAD)", ylabel="Contribution SHAP du montant")
    ax.axhline(0, linewidth=0.7, color="#888888")
    fig.colorbar(nuage, ax=ax, label="Nuit", ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(REPORTS / "shap_montant.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ctx = contexte()
    expliquer_global(ctx, shap.TreeExplainer(ctx.m))
