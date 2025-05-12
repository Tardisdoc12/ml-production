# Cadrage du projet

## Entrée
- Type : texte (.txt)
exemple "you are so dumb, you are such a failure for your parents"

## Sortie
- Label binaire (0/1) : Harcelement / Non Hacelement (raciste/xenophobie/sur la base de religion)
exemple => "harcelement" on empêche le tweet d'être publié.
au bout de trois tweets considéré harcelement, on ban ip la personne.

## Métriques
- Classification : Accuracy, F1‑score

## Contraintes
- Latence < 200 ms par requête
- Taille modèle < 100 Mo
- Budget GPU : T4 max