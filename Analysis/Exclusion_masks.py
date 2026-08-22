import numpy as np

def filter_exclusion_zones(land_cover_data, protected_areas_data=None, urban_data=None):
    """
    Creează o mască de excludere pentru a elimina din start zonele ilegale sau sensibile:
    - Arii protejate / Natura 2000
    - Zone urbane / Intravilan
    - Terenuri agricole fertile / Păduri / Ape (bazat pe codurile de utilizare a terenului)

    Parametri:
    - land_cover_data: Raster cu tipul de utilizare a terenului (ex: Corine Land Cover, unde fiecare număr reprezintă o clasă).
    - protected_areas_data: Matrice binară unde 1 = arie protejată, 0 = în afara ei.
    - urban_data: Matrice binară unde 1 = zonă urbană, 0 = în afara ei.

    Returnează:
    - O mască booleană unde True = zonă permisă pentru construcție, False = zonă interzisă.
    """
    # Pornim de la premisa că tot terenul este inițial permis (True)
    allowed_mask = np.ones(land_cover_data.shape, dtype=bool)

    # 1. Excludem ariile protejate (dacă avem harta)
    if protected_areas_data is not None:
        no_protected_areas = (protected_areas_data == 0)
        allowed_mask = np.logical_and(allowed_mask, no_protected_areas)

    # 2. Excludem zonele urbane / construite dens
    if urban_data is not None:
        no_urban_areas = (urban_data == 0)
        allowed_mask = np.logical_and(allowed_mask, no_urban_areas)

    # 3. Excludem clasele interzise din harta de utilizare a terenului (Land Cover)
    # Exemplu de coduri standard în hărțile de tip Corine Land Cover:
    # - Păduri = Cod 3xx (ex: 311, 312, 313)
    # - Ape / Lacuri / Râuri = Cod 5xx (ex: 511, 512)
    # - Terenuri arabile extrem de fertile protejate = Coduri specifice stabilite de voi

    forbidden_classes = [311, 312, 313, 511, 512, 111, 112] # Coduri de exemplu pentru păduri, ape, orașe

    for code in forbidden_classes:
        is_forbidden = (land_cover_data == code)
        # Transformăm în False zonele care conțin aceste clase interzise
        allowed_mask = np.logical_and(allowed_mask, ~is_forbidden)

    return allowed_mask