import os
import glob

import numpy as np
import rasterio


# ============================================================
# PATH-URI
# ============================================================

ANALYSIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    ANALYSIS_DIR
)


SEARCH_FOLDERS = [
    os.path.join(
        PROJECT_ROOT,
        "Toate fisierele"
    ),

    os.path.join(
        PROJECT_ROOT,
        "LandCover"
    )
]


# ============================================================
# FUNCȚIE ANALIZĂ RASTER
# ============================================================

def analyze_raster(path):

    print()
    print("=" * 80)
    print(os.path.basename(path))
    print("=" * 80)

    try:

        with rasterio.open(path) as src:

            data = src.read(
                1,
                masked=True
            )

            print(
                "PATH:",
                path
            )

            print(
                "CRS:",
                src.crs
            )

            print(
                "DIMENSIUNE:",
                src.width,
                "x",
                src.height
            )

            print(
                "REZOLUȚIE:",
                src.res
            )

            print(
                "DTYPE:",
                src.dtypes[0]
            )

            print(
                "NODATA:",
                src.nodata
            )

            print(
                "BOUNDS:",
                src.bounds
            )

            # ------------------------------------------------
            # Scoatem pixelii mascați
            # ------------------------------------------------

            values_data = data.compressed()

            if values_data.size == 0:

                print(
                    "!!! NU EXISTĂ PIXELI VALIZI !!!"
                )

                return

            # ------------------------------------------------
            # Valori unice
            # ------------------------------------------------

            values, counts = np.unique(
                values_data,
                return_counts=True
            )

            print()
            print(
                "NUMĂR VALORI UNICE:",
                len(values)
            )

            print(
                "MIN:",
                values.min()
            )

            print(
                "MAX:",
                values.max()
            )

            print(
                "MEAN:",
                float(
                    np.mean(values_data)
                )
            )

            print()
            print(
                "PRIMELE VALORI:"
            )

            for value, count in zip(
                    values[:50],
                    counts[:50]
            ):

                percentage = (
                        count /
                        values_data.size *
                        100.0
                )

                print(
                    f"  {value} "
                    f"-> {count} pixeli "
                    f"({percentage:.3f}%)"
                )

            # ------------------------------------------------
            # DIAGNOSTIC AUTOMAT
            # ------------------------------------------------

            print()
            print("-" * 80)
            print("DIAGNOSTIC")
            print("-" * 80)

            unique_count = len(
                values
            )

            max_value = float(
                values.max()
            )

            min_value = float(
                values.min()
            )

            integer_like = np.all(
                np.abs(
                    values.astype(np.float64)
                    -
                    np.round(
                        values.astype(
                            np.float64
                        )
                    )
                )
                <
                1e-6
            )

            # ================================================
            # POSIBIL SENTINEL-2 SCL
            # ================================================

            if (
                    integer_like
                    and
                    min_value >= 0
                    and
                    max_value <= 11
            ):

                print(
                    ">>> POSIBIL SENTINEL-2 "
                    "SCENE CLASSIFICATION (SCL)"
                )

                print(
                    ">>> Are clase 0-11."
                )

            # ================================================
            # POSIBIL CLCPLUS
            # ================================================

            elif (
                    integer_like
                    and
                    min_value >= 0
                    and
                    max_value <= 255
                    and
                    unique_count <= 50
            ):

                print(
                    ">>> POSIBIL RASTER "
                    "CATEGORIC LAND COVER."
                )

                print(
                    ">>> Acesta merită verificat "
                    "pentru utilizarea în MCDA."
                )

            # ================================================
            # POSIBIL CORINE
            # ================================================

            elif (
                    integer_like
                    and
                    max_value <= 999
                    and
                    unique_count <= 100
            ):

                print(
                    ">>> POSIBIL CORINE "
                    "LAND COVER."
                )

                print(
                    ">>> Clasele de tip "
                    "111 / 211 / 311 etc. "
                    "pot fi utilizabile."
                )

            # ================================================
            # RASTER CONTINUU / ALT PRODUS
            # ================================================

            else:

                print(
                    ">>> NU PARE O HARTĂ "
                    "CATEGORICĂ LAND COVER."
                )

                print(
                    ">>> Nu o folosim pentru "
                    "excluderi până nu știm "
                    "ce reprezintă valorile."
                )

            print("-" * 80)

    except Exception as exc:

        print(
            "EROARE:",
            repr(exc)
        )


# ============================================================
# CĂUTĂM FIȘIERELE
# ============================================================

files = []

for folder in SEARCH_FOLDERS:

    if not os.path.isdir(folder):
        continue

    files.extend(
        glob.glob(
            os.path.join(
                folder,
                "*.tif"
            )
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                folder,
                "*.tiff"
            )
        )
    )


# ============================================================
# FILTRĂM FIȘIERE INTERESANTE
# ============================================================

interesting = []

for path in files:

    name = os.path.basename(
        path
    ).lower()

    if (
            "landcover" in name
            or
            "classification" in name
            or
            "clms" in name
            or
            "clc" in name
    ):

        interesting.append(
            path
        )


# Eliminăm duplicatele
interesting = list(
    dict.fromkeys(
        interesting
    )
)


# ============================================================
# RUN
# ============================================================

print()
print("#" * 80)
print("VERIFICARE TOATE RASTERELE LAND COVER")
print("#" * 80)

print(
    "Fișiere găsite:",
    len(interesting)
)


if not interesting:

    print(
        "Nu am găsit rastere."
    )


for path in interesting:

    analyze_raster(
        path
    )


print()
print("#" * 80)
print("FINAL")
print("#" * 80)