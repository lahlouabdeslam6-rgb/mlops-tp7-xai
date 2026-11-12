"""Compare MDI et baisse du F1 par permutation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from xai_commun import REGLAGES, REPORTS, contexte, enregistrer


def importance(ctx):
    resultats = permutation_importance(
        ctx.m, ctx.Xte, ctx.yte, scoring="f1",
        n_repeats=REGLAGES["permutation_n_repeats"], random_state=REGLAGES["permutation_random_state"]
    )
    table = pd.DataFrame({
        "mdi": ctx.m.feature_importances_,
        "permutation": resultats.importances_mean,
        "ecart_type": resultats.importances_std,
    }, index=ctx.Xte.columns).sort_values("permutation", ascending=False)
    enregistrer("importance.json", table.to_dict(orient="index"))
    ax = table.permutation.plot.barh(xerr=table.ecart_type, color="#4592ef")
    ax.invert_yaxis()
    ax.set_xlabel("Perte de F1 après permutation de la variable")
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS / "permutation.png", dpi=140)
    plt.close(ax.figure)


if __name__ == "__main__":
    importance(contexte())
