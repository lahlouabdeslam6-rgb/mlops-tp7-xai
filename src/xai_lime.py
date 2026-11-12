"""Mesure la fidélité et la variabilité des explications locales LIME."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from xai_commun import CAS, REGLAGES, REPORTS, VARIABLES, contexte, enregistrer


def expliquer_lime(ctx):
    def predire(valeurs):
        return ctx.m.predict_proba(pd.DataFrame(valeurs, columns=VARIABLES))

    voisins_reference = REGLAGES["lime_num_samples"]
    graine_reference = REGLAGES["lime_random_state"]
    resultats = {}
    REPORTS.mkdir(exist_ok=True)
    for voisins in (500, voisins_reference):
        essais = []
        for graine in range(5):
            explainer = LimeTabularExplainer(
                ctx.Xtr.to_numpy(), feature_names=VARIABLES,
                class_names=["fraude" if c == 1 else "normal" for c in ctx.m.classes_],
                mode="classification", random_state=graine,
            )
            exp = explainer.explain_instance(
                CAS.to_numpy()[0], predire, labels=(ctx.classe_fraude,),
                num_features=len(VARIABLES), num_samples=voisins,
            )
            essais.append({
                "graine": graine, "r2_local": float(exp.score),
                "poids": {VARIABLES[i]: float(v) for i, v in exp.as_map()[ctx.classe_fraude]},
            })
            if voisins == voisins_reference and graine == graine_reference:
                exp.save_to_file(str(REPORTS / "lime_benali.html"), labels=(ctx.classe_fraude,))
                fig = exp.as_pyplot_figure(label=ctx.classe_fraude)
                fig.tight_layout()
                fig.savefig(REPORTS / "lime_benali.png", dpi=140, bbox_inches="tight")
                plt.close(fig)
                resultats["regles_reference"] = dict(exp.as_list(label=ctx.classe_fraude))
        stabilite = {}
        for nom in VARIABLES:
            poids = [essai["poids"][nom] for essai in essais]
            stabilite[nom] = {
                "ecart_type": float(np.std(poids)),
                "signe_constant": bool(np.all(np.sign(poids) == np.sign(poids[0]))),
            }
        resultats[str(voisins)] = {"essais": essais, "stabilite": stabilite}
    return enregistrer("lime_stabilite.json", resultats)


if __name__ == "__main__":
    expliquer_lime(contexte())
