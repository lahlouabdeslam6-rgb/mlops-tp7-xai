"""Compare la forêt aux modèles lisibles sur le même test."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from xai_commun import CAS, VARIABLES, contexte, enregistrer


def avec_log_montant(table):
    table = table.copy()
    table["log_montant"] = np.log10(table.pop("montant"))
    return table


def comparer(ctx):
    resultats = {"fraudes_test": int(ctx.yte.sum()), "modeles": {}}

    def evaluer(nom, modele, test):
        pred = modele.predict(test)
        resultats["modeles"][nom] = {
            "f1": f1_score(ctx.yte, pred), "rappel": recall_score(ctx.yte, pred),
            "fraudes_detectees": int(((pred == 1) & (ctx.yte == 1)).sum()),
        }

    evaluer("foret", ctx.m, ctx.Xte)
    for nom, transformer in (("logistique_brut", lambda table: table),
                              ("logistique_log10", avec_log_montant)):
        train = transformer(ctx.Xtr)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr.fit(train, ctx.ytr)
        evaluer(nom, lr, transformer(ctx.Xte))
    resultats["coefficients_standardises_log10"] = dict(zip(train.columns, lr[-1].coef_[0].tolist()))
    resultats["rapport_de_cotes_par_unite"] = dict(zip(
        train.columns, np.exp(lr[-1].coef_[0] / lr[0].scale_).tolist()
    ))
    for profondeur in (2, 3, 6, None):
        arbre = DecisionTreeClassifier(max_depth=profondeur, random_state=0).fit(ctx.Xtr, ctx.ytr)
        nom = f"arbre_{profondeur}" if profondeur is not None else "arbre_sans_limite"
        evaluer(nom, arbre, ctx.Xte)
        resultats["modeles"][nom]["feuilles"] = int(arbre.get_n_leaves())
        if profondeur == 3:
            resultats["regles_arbre_3"] = export_text(arbre, feature_names=VARIABLES, decimals=1)
            resultats["benali_arbre_3"] = {
                "feuille": int(arbre.apply(CAS)[0]), "proba": float(arbre.predict_proba(CAS)[0, 1])
            }
    return enregistrer("interpretables.json", resultats)


if __name__ == "__main__":
    comparer(contexte())
