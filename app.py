import os
import traceback

import numpy as np
import rasterio

from rasterio.mask import mask
from rasterio.warp import reproject
from rasterio.enums import Resampling
from pyproj import Transformer, Geod

from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATH-URI
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)


def find_raster_path():
    """
    Caută automat rasterul rezultat_suitabilitate.tif
    în locațiile uzuale ale proiectului.
    """

    candidates = [
        os.path.join(
            BACKEND_DIR,
            "rezultat_suitabilitate.tif"
        ),

        os.path.join(
            PROJECT_ROOT,
            "rezultat_suitabilitate.tif"
        ),

        os.path.join(
            PROJECT_ROOT,
            "Analysis",
            "rezultat_suitabilitate.tif"
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # dacă nu îl găsim, păstrăm prima cale pentru mesajul de eroare
    return candidates[0]


RASTER_PATH = find_raster_path()


def find_landcover_path():
    """
    Caută un raster Land Cover REAL, separat de rasterul MCDA.

    Ordinea de preferință:
    1. LANDCOVER_PATH din variabila de mediu;
    2. CLCplus Backbone / ESA WorldCover / CORINE;
    3. abia la final un fișier generic LandCover*.tif.

    IMPORTANT:
    Banda 6 din rezultat_suitabilitate.tif NU mai este folosită
    ca Land Cover.
    """

    env_path = os.environ.get(
        "LANDCOVER_PATH"
    )

    if env_path:

        env_path = os.path.abspath(
            env_path
        )

        if os.path.exists(
                env_path
        ):

            return env_path

    roots = [
        BACKEND_DIR,
        PROJECT_ROOT,
        os.path.join(
            PROJECT_ROOT,
            "Analysis"
        ),
        os.path.join(
            PROJECT_ROOT,
            "Toate fisierele"
        ),
        os.path.join(
            PROJECT_ROOT,
            "LandCover"
        ),
        os.path.join(
            PROJECT_ROOT,
            "Data"
        ),
    ]

    # --------------------------------------------------------
    # 1. NUME EXPLICITE / SIGURE
    # --------------------------------------------------------

    preferred_exact_names = [
        "CLCplus_Backbone_2023.tif",
        "CLCplus_Backbone.tif",
        "WorldCover.tif",
        "ESA_WorldCover.tif",
        "ESA_WorldCover_10m.tif",
        "CORINE_LandCover.tif",
        "CORINE.tif",
    ]

    for filename in preferred_exact_names:

        for root in roots:

            path = os.path.join(
                root,
                filename
            )

            if os.path.exists(
                    path
            ):

                return path

    # --------------------------------------------------------
    # 2. CĂUTARE RECURSIVĂ DUPĂ NUME EXPLICIT
    # --------------------------------------------------------

    preferred_candidates = []
    generic_candidates = []

    for root in roots:

        if not os.path.isdir(
                root
        ):

            continue

        for current_root, _, files in os.walk(
                root
        ):

            for filename in files:

                lower = filename.lower()

                if not lower.endswith(
                        (".tif", ".tiff")
                ):

                    continue

                # Layere HRL Croplands de fenologie / pattern.
                # NU sunt Land Cover categorial.
                if any(
                        token in lower
                        for token in (
                                "cpmce",
                                "cpmch",
                                "cpmcd",
                                "cpmcecl",
                                "cpmchcl",
                                "ctycl",
                                "cpbs",
                                "cpsc",
                                "cpfl",
                        )
                ):

                    continue

                priority = None

                if (
                        "clcplus" in lower
                        or
                        "backbone" in lower
                ):

                    priority = 0

                elif (
                        "worldcover" in lower
                        or
                        "esa_world" in lower
                ):

                    priority = 1

                elif (
                        "corine" in lower
                        or
                        "u2018_clc" in lower
                        or
                        "u2012_clc" in lower
                        or
                        "u2006_clc" in lower
                ):

                    priority = 2

                if priority is not None:

                    preferred_candidates.append(
                        (
                            priority,
                            os.path.join(
                                current_root,
                                filename
                            )
                        )
                    )

                elif "landcover" in lower:

                    generic_candidates.append(
                        os.path.join(
                            current_root,
                            filename
                        )
                    )

    if preferred_candidates:

        preferred_candidates.sort(
            key=lambda item: (
                item[0],
                len(item[1]),
                item[1].lower()
            )
        )

        return preferred_candidates[0][1]

    # --------------------------------------------------------
    # 3. FALLBACK GENERIC
    # --------------------------------------------------------
    # Îl acceptăm doar ca fișier candidat.
    # infer_landcover_allowed_mask() îi validează codurile și
    # îl respinge dacă este de fapt CPMCE / alt produs greșit.
    # --------------------------------------------------------

    if generic_candidates:

        generic_candidates.sort(
            key=lambda path: (
                len(path),
                path.lower()
            )
        )

        return generic_candidates[0]

    return os.path.join(
        PROJECT_ROOT,
        "Data",
        "CLCplus_Backbone_2023.tif"
    )

# ============================================================
# LAND COVER CLUJ - FIȘIERUL CORECT PREGĂTIT DIN CLCplus 2023
# ============================================================
#
# Folosim explicit rasterul creat din tile-urile E53N26 + E53N27,
# reproiectat în EPSG:3844 cu Resampling.nearest.
# Astfel backend-ul NU mai poate selecta accidental vechiul CPMCE
# sau un alt fișier numit generic LandCover.tif.
#
LANDCOVER_PATH = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "LandCover_Cluj_3844.tif"
)

# Rasterul este CLCplus Backbone 2023 RAS, deci legenda este fixă.
# Nu mai folosim detectare automată pentru acest proiect.
LANDCOVER_SCHEME = "clcplus"


# ============================================================
# NATURA 2000 CLUJ - MASCĂ BINARĂ 0/1
# ============================================================
#
# Raster creat din layerul NaturaSite_polygon și aliniat exact
# pe gridul LandCover_Cluj_3844.tif:
#   0 = în afara unui sit Natura 2000
#   1 = în interiorul unui sit Natura 2000
#
PROTECTED_AREAS_PATH = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "ProtectedAreas_Natura2000_Cluj_3844.tif"
)


# ============================================================
# CONFIGURARE ANALIZĂ
# ============================================================

SCORE_THRESHOLD = 0.35


# ============================================================
# LAND COVER - LEGENDE SUPORTATE
# ============================================================

# ------------------------------------------------------------
# CLCplus Backbone 2023 (10 m)
# ------------------------------------------------------------
# 1  = Sealed
# 2  = Woody needle leaved trees
# 3  = Woody broadleaved deciduous trees
# 4  = Woody broadleaved evergreen trees
# 5  = Low-growing woody plants
# 6  = Permanent herbaceous
# 7  = Periodically herbaceous
# 8  = Lichens and mosses
# 9  = Non- and sparsely vegetated
# 10 = Water
# 11 = Snow and ice
# 253 = Coastal seawater buffer
# 254 = Outside area
# 255 = NoData

CLCPLUS_CLASSES = {
    1: "Sealed",
    2: "Woody needle leaved trees",
    3: "Woody broadleaved deciduous trees",
    4: "Woody broadleaved evergreen trees",
    5: "Low-growing woody plants",
    6: "Permanent herbaceous",
    7: "Periodically herbaceous",
    8: "Lichens and mosses",
    9: "Non- and sparsely vegetated",
    10: "Water",
    11: "Snow and ice",
}

# Excluderi tehnice HARD pentru amplasare:
# suprafețe sigilate, arbori/pădure, apă, zăpadă/gheață.
FORBIDDEN_CLCPLUS_CLASSES = {
    1,
    2,
    3,
    4,
    10,
    11,
}


# ------------------------------------------------------------
# ESA WorldCover (10 m)
# ------------------------------------------------------------

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

FORBIDDEN_WORLDCOVER_CLASSES = {
    10,   # Tree cover
    50,   # Built-up
    70,   # Snow and ice
    80,   # Water
    90,   # Wetland
    95,   # Mangroves
}


# ------------------------------------------------------------
# CORINE Land Cover
# ------------------------------------------------------------

CORINE_CLASSES = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",

    121: "Industrial or commercial units",
    122: "Road and rail networks",
    123: "Port areas",
    124: "Airports",

    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",

    141: "Green urban areas",
    142: "Sport and leisure facilities",

    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",

    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",

    231: "Pastures",

    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture",
    244: "Agro-forestry areas",

    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",

    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",

    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",

    411: "Inland marshes",
    412: "Peat bogs",

    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",

    511: "Water courses",
    512: "Water bodies",

    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean",
}

FORBIDDEN_CORINE_CLASSES = {
    # artificial / urban / infrastructură
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    132,
    133,
    141,
    142,

    # forest
    311,
    312,
    313,

    # snow / ice
    335,

    # wetlands
    411,
    412,
    421,
    422,
    423,

    # water
    511,
    512,
    521,
    522,
    523,
}


# ============================================================
# GEODEZIE
# ============================================================

GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# FUNCȚII UTILE
# ============================================================

def close_ring(coords):
    """
    Se asigură că poligonul GeoJSON este închis.
    """

    coords = [
        (
            float(lon),
            float(lat)
        )
        for lon, lat in coords
    ]

    if len(coords) < 3:
        raise ValueError(
            "Poligonul trebuie să aibă "
            "cel puțin 3 puncte."
        )

    if coords[0] != coords[-1]:
        coords.append(
            coords[0]
        )

    return coords


# ============================================================

def geodesic_area_ha(coords_wgs84):
    """
    Calculează suprafața REALĂ a poligonului pe elipsoidul WGS84.

    Rezultatul este în hectare.
    """

    coords = close_ring(
        coords_wgs84
    )

    lons = [
        p[0]
        for p in coords
    ]

    lats = [
        p[1]
        for p in coords
    ]

    area_m2, _ = (
        GEOD.polygon_area_perimeter(
            lons,
            lats
        )
    )

    area_m2 = abs(
        float(area_m2)
    )

    return (
            area_m2 /
            10000.0
    )


# ============================================================

def transform_polygon(
        coords_wgs84,
        dst_crs
):
    """
    Transformă poligonul desenat din EPSG:4326
    în CRS-ul rasterului.
    """

    coords = close_ring(
        coords_wgs84
    )

    if dst_crs is None:
        raise ValueError(
            "Rasterul nu are CRS definit."
        )

    if dst_crs.to_epsg() == 4326:

        transformed = coords

    else:

        transformer = Transformer.from_crs(
            "EPSG:4326",
            dst_crs,
            always_xy=True
        )

        transformed = [
            transformer.transform(
                lon,
                lat
            )
            for lon, lat in coords
        ]

    return [{
        "type": "Polygon",
        "coordinates": [
            transformed
        ]
    }]


# ============================================================

def load_aligned_landcover(
        dst_crs,
        dst_transform,
        dst_shape
):
    """
    Citește rasterul Land Cover SEPARAT și îl aliniază exact
    pe grila crop-ului din rezultat_suitabilitate.tif.

    Resampling.nearest este obligatoriu pentru date categoriale.
    """

    if not os.path.exists(
            LANDCOVER_PATH
    ):

        raise FileNotFoundError(
            "\nNu am găsit un raster Land Cover real.\n"
            f"Calea căutată: {LANDCOVER_PATH}\n\n"
            "Adaugă un CLCplus Backbone 2023 / ESA WorldCover / "
            "CORINE GeoTIFF și setează LANDCOVER_PATH dacă este "
            "în alt folder."
        )

    destination = np.full(
        dst_shape,
        np.nan,
        dtype=np.float64
    )

    with rasterio.open(
            LANDCOVER_PATH
    ) as lc_src:

        if lc_src.count < 1:

            raise ValueError(
                "Rasterul Land Cover nu conține nicio bandă."
            )

        if lc_src.crs is None:

            raise ValueError(
                "Rasterul Land Cover nu are CRS definit."
            )

        reproject(
            source=rasterio.band(
                lc_src,
                1
            ),
            destination=destination,
            src_transform=lc_src.transform,
            src_crs=lc_src.crs,
            src_nodata=lc_src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            init_dest_nodata=True
        )

    return destination


# ============================================================

def sample_landcover_wgs84(
        lon,
        lat
):
    """
    Citește valoarea Land Cover la un punct WGS84 din rasterul
    Land Cover separat.
    """

    if not os.path.exists(
            LANDCOVER_PATH
    ):

        raise FileNotFoundError(
            "Raster Land Cover lipsă: "
            f"{LANDCOVER_PATH}"
        )

    with rasterio.open(
            LANDCOVER_PATH
    ) as lc_src:

        if lc_src.crs is None:

            raise ValueError(
                "Rasterul Land Cover nu are CRS."
            )

        if lc_src.crs.to_epsg() != 4326:

            transformer = Transformer.from_crs(
                "EPSG:4326",
                lc_src.crs,
                always_xy=True
            )

            x, y = transformer.transform(
                lon,
                lat
            )

        else:

            x = lon
            y = lat

        value = list(
            lc_src.sample(
                [(x, y)]
            )
        )[0][0]

        if (
                lc_src.nodata is not None
                and
                value == lc_src.nodata
        ):
            return np.nan

        try:
            return float(
                value
            )
        except Exception:
            return np.nan


# ============================================================

def load_aligned_binary_raster(
        source_path,
        dst_crs,
        dst_transform,
        dst_shape
):
    """
    Citește un raster binar 0/1 și îl aliniază exact la crop-ul
    curent din rasterul MCDA.

    Este folosit pentru Natura 2000.
    Resampling.nearest este obligatoriu pentru o mască categorială.

    Returnează float64:
        0.0 = în afara zonei protejate
        1.0 = în zona protejată
        NaN = lipsă acoperire / NoData
    """

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            "Raster binar lipsă: "
            f"{source_path}"
        )

    destination = np.full(
        dst_shape,
        np.nan,
        dtype=np.float64
    )

    with rasterio.open(source_path) as src:

        if src.count < 1:
            raise ValueError(
                f"Rasterul {source_path} nu conține nicio bandă."
            )

        if src.crs is None:
            raise ValueError(
                f"Rasterul {source_path} nu are CRS definit."
            )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            init_dest_nodata=True
        )

    return destination


# ============================================================

def sample_binary_raster_wgs84(
        source_path,
        lon,
        lat
):
    """
    Citește un raster binar la un punct WGS84.

    Return:
        0.0 / 1.0 sau np.nan dacă punctul este în afara acoperirii.
    """

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            "Raster binar lipsă: "
            f"{source_path}"
        )

    with rasterio.open(source_path) as src:

        if src.crs is None:
            raise ValueError(
                f"Rasterul {source_path} nu are CRS."
            )

        if src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(
                "EPSG:4326",
                src.crs,
                always_xy=True
            )
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat

        # Evităm ca rasterio.sample să întoarcă o valoare implicită
        # pentru un punct aflat în afara rasterului.
        bounds = src.bounds
        if not (
                bounds.left <= x <= bounds.right
                and
                bounds.bottom <= y <= bounds.top
        ):
            return np.nan

        value = list(src.sample([(x, y)]))[0][0]

        if (
                src.nodata is not None
                and
                value == src.nodata
        ):
            return np.nan

        try:
            return float(value)
        except Exception:
            return np.nan


# ============================================================

def safe_mean(
        data,
        valid_mask
):
    """
    Medie doar din valorile valide.
    """

    values = np.asarray(
        data,
        dtype=np.float64
    )[valid_mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return None

    return float(
        np.mean(values)
    )


# ============================================================

def circular_mean_degrees(
        aspect_data,
        valid_mask,
        slope_data=None
):
    """
    Calculează corect media orientării.

    Exemplu:
    359° + 1° -> media ~0°
    nu 180°.

    Eliminăm și pixelii aproape plați,
    deoarece aspectul lor nu este relevant.
    """

    aspect = np.asarray(
        aspect_data,
        dtype=np.float64
    )

    mask_ok = (
            valid_mask
            &
            np.isfinite(aspect)
            &
            (aspect >= 0.0)
            &
            (aspect <= 360.0)
    )

    # Pentru teren aproape plat orientarea este instabilă.
    if slope_data is not None:

        slope = np.asarray(
            slope_data,
            dtype=np.float64
        )

        mask_ok &= (
                np.isfinite(slope)
                &
                (slope >= 2.0)
        )

    values = aspect[
        mask_ok
    ]

    if values.size == 0:
        return None

    radians = np.deg2rad(
        values
    )

    mean_sin = np.mean(
        np.sin(radians)
    )

    mean_cos = np.mean(
        np.cos(radians)
    )

    strength = np.hypot(
        mean_sin,
        mean_cos
    )

    # Dacă nu există direcție dominantă
    if strength < 1e-8:
        return None

    angle = np.degrees(
        np.arctan2(
            mean_sin,
            mean_cos
        )
    )

    return float(
        (angle + 360.0) % 360.0
    )


# ============================================================

def infer_landcover_allowed_mask(
        lc_data,
        valid_mask,
        source_path=None,
        scheme="auto"
):
    """
    Interpretează un raster Land Cover REAL.

    Sunt suportate:
        - CLCplus Backbone 2023
        - ESA WorldCover
        - CORINE Land Cover

    IMPORTANT:
    O bandă care conține doar 0/1 NU mai este acceptată automat.
    Asta previne exact situația în care o bandă Land Cover
    stricată, plină cu 0, face pădurea să pară permisă.

    Returnează:
        allowed_mask
        mod_detectat
        pixeli_excluși_hard
        pixeli_fără_landcover
        class_summary
    """

    lc = np.asarray(
        lc_data,
        dtype=np.float64
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool
    )

    if lc.shape != valid_mask.shape:

        raise ValueError(
            "Land Cover și valid_mask nu au aceeași dimensiune."
        )

    finite = (
            valid_mask
            &
            np.isfinite(
                lc
            )
    )

    values = lc[
        finite
    ]

    if values.size == 0:

        raise ValueError(
            "Nu există date Land Cover valide în parcela analizată."
        )

    # Land Cover este categorial. Valori zecimale indică de obicei
    # reproiectare cu bilinear/cubic, ceea ce este greșit.
    rounded = np.rint(
        values
    )

    if not np.all(
            np.abs(
                values - rounded
            ) < 1e-6
    ):

        raise ValueError(
            "\nRasterul Land Cover are valori zecimale.\n"
            "Pentru Land Cover trebuie folosit "
            "Resampling.nearest, nu bilinear/cubic."
        )

    unique_values = set(
        rounded.astype(
            np.int64
        ).tolist()
    )

    source_name = ""

    if source_path:

        source_name = os.path.basename(
            source_path
        ).lower()

    requested_scheme = (
        scheme
        if scheme is not None
        else "auto"
    )

    requested_scheme = str(
        requested_scheme
    ).strip().lower()

    # --------------------------------------------------------
    # Detectare fișier HRL Croplands greșit.
    # Valorile 6552x / 6553x sunt caracteristice valorilor
    # speciale ale mai multor layere HRL Croplands.
    # --------------------------------------------------------

    if any(
            value >= 65000
            for value in unique_values
    ):

        raise ValueError(
            "\nFIȘIER LAND COVER GREȘIT.\n"
            "Rasterul conține coduri de tip 655xx și nu este "
            "un raster categorial de Land Cover utilizabil aici.\n\n"
            "Dacă ai folosit fișierul CLMS ...CPMCE..., acela "
            "reprezintă data de emergență a culturii principale, "
            "nu clasa de acoperire a terenului.\n\n"
            "Folosește CLCplus Backbone 2023 RAS (10 m), "
            "ESA WorldCover sau CORINE Land Cover."
        )

    # --------------------------------------------------------
    # Detectare pe baza numelui fișierului / config
    # --------------------------------------------------------

    detected_scheme = None

    if requested_scheme in {
        "clcplus",
        "worldcover",
        "corine"
    }:

        detected_scheme = (
            requested_scheme
        )

    elif requested_scheme != "auto":

        raise ValueError(
            "LANDCOVER_SCHEME trebuie să fie "
            "auto, clcplus, worldcover sau corine."
        )

    if detected_scheme is None:

        if (
                "clcplus" in source_name
                or
                "backbone" in source_name
        ):

            detected_scheme = "clcplus"

        elif (
                "worldcover" in source_name
                or
                "esa_world" in source_name
        ):

            detected_scheme = "worldcover"

        elif (
                "corine" in source_name
                or
                "u2018_clc" in source_name
                or
                "u2012_clc" in source_name
        ):

            detected_scheme = "corine"

    # --------------------------------------------------------
    # Detectare după codurile tematice
    # --------------------------------------------------------

    if detected_scheme is None:

        worldcover_codes = set(
            WORLDCOVER_CLASSES.keys()
        )

        clcplus_codes = set(
            CLCPLUS_CLASSES.keys()
        ) | {
                            253,
                            254,
                            255
                        }

        corine_codes = set(
            CORINE_CLASSES.keys()
        )

        # WorldCover are o semnătură destul de distinctă dacă
        # apar coduri 20/30/40/.../100.
        if (
                unique_values.issubset(
                    worldcover_codes | {0}
                )
                and
                any(
                    value in {
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        95,
                        100
                    }
                    for value in unique_values
                )
        ):

            detected_scheme = "worldcover"

        elif (
                unique_values.issubset(
                    clcplus_codes | {0}
                )
                and
                any(
                    value in {
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        11,
                        253,
                        254,
                        255
                    }
                    for value in unique_values
                )
        ):

            detected_scheme = "clcplus"

        elif (
                len(
                    unique_values
                    &
                    corine_codes
                ) > 0
                and
                unique_values.issubset(
                    corine_codes | {0, 255}
                )
        ):

            detected_scheme = "corine"

    # --------------------------------------------------------
    # Nu mai ghicim 0/1.
    # --------------------------------------------------------

    if detected_scheme is None:

        raise ValueError(
            "\nNu pot identifica sigur legenda Land Cover.\n"
            f"Valori găsite în parcelă: "
            f"{sorted(unique_values)[:50]}\n"
            f"Fișier: {source_path}\n\n"
            "Dacă rasterul este CLCplus / WorldCover / CORINE, "
            "setează variabila LANDCOVER_SCHEME corespunzător.\n"
            "O bandă doar cu 0/1 nu este acceptată automat, "
            "pentru a evita rezultate fals pozitive."
        )

    # --------------------------------------------------------
    # Configurația fiecărei legende.
    # --------------------------------------------------------

    if detected_scheme == "clcplus":

        classes = CLCPLUS_CLASSES
        forbidden_classes = (
            FORBIDDEN_CLCPLUS_CLASSES
        )

        explicit_nodata = {
            0,
            253,
            254,
            255,
        }

        mode_label = (
            "CLCplus Backbone"
        )

    elif detected_scheme == "worldcover":

        classes = WORLDCOVER_CLASSES
        forbidden_classes = (
            FORBIDDEN_WORLDCOVER_CLASSES
        )

        explicit_nodata = {
            0,
            255,
        }

        mode_label = (
            "ESA WorldCover"
        )

    else:

        classes = CORINE_CLASSES
        forbidden_classes = (
            FORBIDDEN_CORINE_CLASSES
        )

        explicit_nodata = {
            0,
            255,
        }

        mode_label = (
            "CORINE Land Cover"
        )

    lc_int = np.full(
        lc.shape,
        -999999,
        dtype=np.int64
    )

    lc_int[
        np.isfinite(
            lc
        )
    ] = np.rint(
        lc[
            np.isfinite(
                lc
            )
        ]
    ).astype(
        np.int64
    )

    thematic_codes = set(
        classes.keys()
    )

    thematic_mask = (
            valid_mask
            &
            np.isfinite(
                lc
            )
            &
            np.isin(
                lc_int,
                list(
                    thematic_codes
                )
            )
    )

    # Valori finite care nu sunt nici clase tematice, nici NoData.
    suspicious_mask = (
            valid_mask
            &
            np.isfinite(
                lc
            )
            &
            ~np.isin(
                lc_int,
                list(
                    thematic_codes
                    |
                    explicit_nodata
                )
            )
    )

    if np.any(
            suspicious_mask
    ):

        suspicious_values = np.unique(
            lc_int[
                suspicious_mask
            ]
        )

        raise ValueError(
            "\nRaster Land Cover incompatibil cu legenda "
            f"{mode_label}.\n"
            "Coduri necunoscute: "
            f"{suspicious_values[:50].tolist()}"
        )

    # Conservator: tot ce nu are o clasă tematică validă
    # rămâne NEPERMIS.
    allowed = np.zeros(
        lc.shape,
        dtype=bool
    )

    forbidden_mask = (
            thematic_mask
            &
            np.isin(
                lc_int,
                list(
                    forbidden_classes
                )
            )
    )

    allowed[
        thematic_mask
    ] = (
        ~forbidden_mask[
            thematic_mask
        ]
    )

    excluded = int(
        np.sum(
            forbidden_mask
        )
    )

    missing_mask = (
            valid_mask
            &
            ~thematic_mask
    )

    missing = int(
        np.sum(
            missing_mask
        )
    )

    class_summary = []

    present_codes, present_counts = np.unique(
        lc_int[
            thematic_mask
        ],
        return_counts=True
    )

    total_valid_pixels = int(
        np.sum(
            valid_mask
        )
    )

    for class_code, class_count in zip(
            present_codes,
            present_counts
    ):

        class_code = int(
            class_code
        )

        class_count = int(
            class_count
        )

        class_summary.append({
            "code":
                class_code,

            "name":
                classes.get(
                    class_code,
                    "Unknown"
                ),

            "pixels":
                class_count,

            "percentage":
                (
                    class_count
                    /
                    total_valid_pixels
                    *
                    100.0
                    if total_valid_pixels > 0
                    else 0.0
                ),

            "forbidden":
                (
                        class_code
                        in forbidden_classes
                )
        })

    return (
        allowed,
        mode_label,
        excluded,
        missing,
        class_summary
    )


# ============================================================

def band_stats(
        name,
        data,
        valid_mask
):
    """
    Afișează statistici în terminal.
    """

    values = np.asarray(
        data,
        dtype=np.float64
    )[valid_mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        print(
            f"[DEBUG] {name}: "
            f"fără valori valide"
        )

        return

    print(
        f"[DEBUG] {name}: "
        f"min={np.min(values):.4f}, "
        f"max={np.max(values):.4f}, "
        f"mean={np.mean(values):.4f}, "
        f"n={values.size}"
    )


# ============================================================
# INFO RASTER LA PORNIRE
# ============================================================

def print_raster_info():

    print("\n")
    print("=" * 72)
    print("HELIO SITE EXPLORER - BACKEND")
    print("=" * 72)

    print(
        "Raster MCDA:",
        RASTER_PATH
    )

    if not os.path.exists(
            RASTER_PATH
    ):

        print(
            "ATENȚIE: rasterul MCDA nu există."
        )

    else:

        try:

            with rasterio.open(
                    RASTER_PATH
            ) as src:

                print(
                    "MCDA CRS:",
                    src.crs
                )

                print(
                    "MCDA număr benzi:",
                    src.count
                )

                print(
                    "MCDA dimensiune:",
                    src.width,
                    "x",
                    src.height
                )

                print(
                    "MCDA rezoluție:",
                    src.res
                )

                print(
                    "MCDA NoData:",
                    src.nodata
                )

                print(
                    "MCDA bounds:",
                    src.bounds
                )

        except Exception as exc:

            print(
                "Nu am putut citi rasterul MCDA:",
                repr(exc)
            )

    print("-" * 72)

    print(
        "Raster Land Cover:",
        LANDCOVER_PATH
    )

    if not os.path.exists(
            LANDCOVER_PATH
    ):

        print(
            "ATENȚIE: rasterul Land Cover real nu există."
        )

        print(
            "Banda 6 din rezultat_suitabilitate.tif "
            "NU va mai fi folosită ca substitut."
        )

    else:

        try:

            with rasterio.open(
                    LANDCOVER_PATH
            ) as lc_src:

                print(
                    "Land Cover CRS:",
                    lc_src.crs
                )

                print(
                    "Land Cover dimensiune:",
                    lc_src.width,
                    "x",
                    lc_src.height
                )

                print(
                    "Land Cover rezoluție:",
                    lc_src.res
                )

                print(
                    "Land Cover dtype:",
                    lc_src.dtypes[0]
                )

                print(
                    "Land Cover NoData:",
                    lc_src.nodata
                )

        except Exception as exc:

            print(
                "Nu am putut citi rasterul Land Cover:",
                repr(exc)
            )

    print(
        "LANDCOVER_SCHEME:",
        LANDCOVER_SCHEME
    )

    print("-" * 72)

    print(
        "Raster Natura 2000:",
        PROTECTED_AREAS_PATH
    )

    if not os.path.exists(PROTECTED_AREAS_PATH):
        print(
            "ATENȚIE: rasterul Natura 2000 nu există."
        )
    else:
        try:
            with rasterio.open(PROTECTED_AREAS_PATH) as pa_src:
                print("Natura 2000 CRS:", pa_src.crs)
                print(
                    "Natura 2000 dimensiune:",
                    pa_src.width,
                    "x",
                    pa_src.height
                )
                print("Natura 2000 rezoluție:", pa_src.res)
                print("Natura 2000 NoData:", pa_src.nodata)
        except Exception as exc:
            print(
                "Nu am putut citi rasterul Natura 2000:",
                repr(exc)
            )

    print(
        "=" * 72
    )

    print()


# ============================================================
# ANALIZĂ POLIGON
# ============================================================

@app.route(
    "/analyze-polygon",
    methods=["POST"]
)
def analyze_polygon():

    data = request.get_json(
        silent=True
    )

    # ========================================================
    # DEBUG REQUEST
    # ========================================================

    print("\n" + "=" * 72)
    print(">>> REQUEST /analyze-polygon <<<")
    print("JSON primit:", data)
    print("=" * 72)

    if (
            not data
            or
            "polygon" not in data
    ):

        print(
            "[EROARE 400] Nu am primit cheia 'polygon'."
        )

        return jsonify({
            "status": "error",
            "message": "Poligon lipsă"
        }), 400

    polygon_coords = data.get(
        "polygon"
    )

    print(
        "[DEBUG] polygon_coords:",
        polygon_coords
    )

    print(
        "[DEBUG] tip polygon:",
        type(polygon_coords)
    )

    if isinstance(
            polygon_coords,
            list
    ):

        print(
            "[DEBUG] număr puncte:",
            len(polygon_coords)
        )

    if (
            not isinstance(
                polygon_coords,
                list
            )
            or
            len(polygon_coords) < 3
    ):

        print(
            "[EROARE 400] Format poligon invalid."
        )

        return jsonify({
            "status": "error",
            "message":
                f"Poligon invalid: {polygon_coords}"
        }), 400

    if not os.path.exists(
            RASTER_PATH
    ):

        return jsonify({
            "status": "error",
            "message":
                f"Fișier MCDA lipsă: "
                f"{RASTER_PATH}"
        }), 500

    if not os.path.exists(
            LANDCOVER_PATH
    ):

        return jsonify({
            "status": "error",
            "message":
                "Fișierul Land Cover real lipsește. "
                "Banda 6 veche nu mai este folosită. "
                f"Calea căutată: {LANDCOVER_PATH}"
        }), 500

    if not os.path.exists(
            PROTECTED_AREAS_PATH
    ):

        return jsonify({
            "status": "error",
            "message":
                "Fișierul Natura 2000 lipsește. "
                f"Calea căutată: {PROTECTED_AREAS_PATH}"
        }), 500

    try:

        # ====================================================
        # SUPRAFAȚA POLIGONULUI
        # ====================================================

        total_area_ha = (
            geodesic_area_ha(
                polygon_coords
            )
        )

        # ====================================================
        # DESCHIDEM RASTERUL MCDA
        # ====================================================

        with rasterio.open(
                RASTER_PATH
        ) as src:

            # Banda 6 NU mai este necesară.
            if src.count < 5:

                raise ValueError(
                    "Rasterul MCDA trebuie să aibă "
                    "minimum 5 benzi: scor, pantă, aspect, "
                    "distanță rețea și iradiere. "
                    f"Are doar {src.count}."
                )

            # =================================================
            # TRANSFORMARE POLIGON
            # =================================================

            geo_polygon = (
                transform_polygon(
                    polygon_coords,
                    src.crs
                )
            )

            # =================================================
            # CROP MCDA
            # =================================================

            out_image, out_transform = mask(
                src,
                geo_polygon,
                crop=True,
                filled=False,
                all_touched=False
            )

            # =================================================
            # BENZI MCDA
            # =================================================

            score = (
                out_image[0]
                .astype(np.float64)
                .filled(np.nan)
            )

            slope = (
                out_image[1]
                .astype(np.float64)
                .filled(np.nan)
            )

            aspect = (
                out_image[2]
                .astype(np.float64)
                .filled(np.nan)
            )

            distance = (
                out_image[3]
                .astype(np.float64)
                .filled(np.nan)
            )

            irradiation = (
                out_image[4]
                .astype(np.float64)
                .filled(np.nan)
            )

            # =================================================
            # LAND COVER REAL, ALINIAT LA CROP-UL MCDA
            # =================================================

            landcover = load_aligned_landcover(
                dst_crs=src.crs,
                dst_transform=out_transform,
                dst_shape=score.shape
            )

            # =================================================
            # NATURA 2000, ALINIAT LA ACELAȘI CROP MCDA
            # =================================================

            protected_areas = load_aligned_binary_raster(
                source_path=PROTECTED_AREAS_PATH,
                dst_crs=src.crs,
                dst_transform=out_transform,
                dst_shape=score.shape
            )

            # =================================================
            # PIXELI VALIZI AI POLIGONULUI
            # =================================================

            score_mask = (
                np.ma.getmaskarray(
                    out_image[0]
                )
            )

            valid_score = (
                    ~score_mask
                    &
                    np.isfinite(score)
                    &
                    (score >= 0.0)
                    &
                    (score <= 1.0)
            )

            if not np.any(
                    valid_score
            ):

                print("\n")
                print("!" * 72)
                print(
                    "[EROARE 400] NU EXISTĂ PIXELI SCORE VALIZI"
                )
                print("!" * 72)

                print(
                    "Raster MCDA:",
                    RASTER_PATH
                )

                print(
                    "Raster CRS:",
                    src.crs
                )

                print(
                    "Raster bounds:",
                    src.bounds
                )

                print(
                    "Raster transform:",
                    src.transform
                )

                print(
                    "Poligon original WGS84:",
                    polygon_coords
                )

                print(
                    "Poligon transformat:",
                    geo_polygon
                )

                print(
                    "Shape crop score:",
                    score.shape
                )

                finite_score_count = int(
                    np.count_nonzero(
                        np.isfinite(score)
                    )
                )

                print(
                    "Valori finite score:",
                    finite_score_count
                )

                print(
                    "Pixeli nemascați în banda score:",
                    int(
                        np.count_nonzero(
                            ~score_mask
                        )
                    )
                )

                if finite_score_count > 0:

                    print(
                        "Score min:",
                        float(
                            np.nanmin(score)
                        )
                    )

                    print(
                        "Score max:",
                        float(
                            np.nanmax(score)
                        )
                    )

                print("!" * 72)
                print()

                return jsonify({
                    "status": "error",
                    "message":
                        "Poligonul nu conține pixeli "
                        "valizi în rasterul MCDA.",
                    "debug": {
                        "raster_crs":
                            str(src.crs),
                        "raster_bounds":
                            [
                                float(src.bounds.left),
                                float(src.bounds.bottom),
                                float(src.bounds.right),
                                float(src.bounds.top)
                            ],
                        "crop_shape":
                            [
                                int(score.shape[0]),
                                int(score.shape[1])
                            ],
                        "finite_score_pixels":
                            finite_score_count
                    }
                }), 400

            # =================================================
            # DEBUG TERMINAL
            # =================================================

            print("\n")
            print("-" * 72)
            print(
                ">>> ANALIZĂ POLIGON <<<"
            )

            print(
                "Suprafață geodezică:",
                round(
                    total_area_ha,
                    4
                ),
                "ha"
            )

            print(
                "Land Cover folosit:",
                LANDCOVER_PATH
            )

            print(
                "Natura 2000 folosit:",
                PROTECTED_AREAS_PATH
            )

            band_stats(
                "SCOR",
                score,
                valid_score
            )

            slope_valid = (
                    valid_score
                    &
                    np.isfinite(slope)
                    &
                    (slope >= 0.0)
                    &
                    (slope <= 90.0)
            )

            band_stats(
                "PANTĂ",
                slope,
                slope_valid
            )

            aspect_valid = (
                    valid_score
                    &
                    np.isfinite(aspect)
                    &
                    (aspect >= 0.0)
                    &
                    (aspect <= 360.0)
            )

            band_stats(
                "ASPECT",
                aspect,
                aspect_valid
            )

            distance_valid = (
                    valid_score
                    &
                    np.isfinite(distance)
                    &
                    (distance >= 0.0)
            )

            band_stats(
                "DISTANȚĂ",
                distance,
                distance_valid
            )

            irradiation_valid = (
                    valid_score
                    &
                    np.isfinite(
                        irradiation
                    )
                    &
                    (irradiation >= 0.0)
            )

            band_stats(
                "IRADIERE",
                irradiation,
                irradiation_valid
            )

            lc_finite = (
                    valid_score
                    &
                    np.isfinite(
                        landcover
                    )
            )

            band_stats(
                "LAND COVER REAL",
                landcover,
                lc_finite
            )

            protected_finite = (
                    valid_score
                    &
                    np.isfinite(protected_areas)
            )

            band_stats(
                "NATURA 2000",
                protected_areas,
                protected_finite
            )

            # =================================================
            # MEDII
            # =================================================

            score_mean = safe_mean(
                score,
                valid_score
            )

            slope_mean = safe_mean(
                slope,
                slope_valid
            )

            aspect_mean = (
                circular_mean_degrees(
                    aspect,
                    valid_score,
                    slope_data=slope
                )
            )

            distance_mean = safe_mean(
                distance,
                distance_valid
            )

            irradiation_mean = (
                safe_mean(
                    irradiation,
                    irradiation_valid
                )
            )

            # =================================================
            # LAND COVER
            # =================================================

            (
                lc_allowed,
                lc_mode,
                lc_excluded_pixels,
                lc_missing_pixels,
                lc_class_summary

            ) = infer_landcover_allowed_mask(
                landcover,
                valid_score,
                source_path=LANDCOVER_PATH,
                scheme=LANDCOVER_SCHEME
            )

            # =================================================
            # NATURA 2000 - RESTRICȚIE HARD
            # =================================================

            protected_known = (
                    valid_score
                    &
                    np.isfinite(protected_areas)
            )

            protected_mask = (
                    protected_known
                    &
                    (protected_areas >= 0.5)
            )

            protected_missing_mask = (
                    valid_score
                    &
                    (~np.isfinite(protected_areas))
            )

            # Dacă nu avem date Natura 2000 pentru un pixel,
            # îl tratăm conservator ca nefezabil.
            protected_allowed = (
                    protected_known
                    &
                    (~protected_mask)
            )

            # =================================================
            # FEZABILITATE
            # =================================================

            feasible_mask = (
                    valid_score
                    &
                    (score >= SCORE_THRESHOLD)
                    &
                    lc_allowed
                    &
                    protected_allowed
            )

            valid_pixels = int(
                np.sum(
                    valid_score
                )
            )

            feasible_pixels = int(
                np.sum(
                    feasible_mask
                )
            )

            hard_landcover_rejected_pixels = int(
                np.sum(
                    valid_score
                    &
                    (~lc_allowed)
                )
            )

            protected_pixels = int(
                np.sum(
                    protected_mask
                )
            )

            protected_missing_pixels = int(
                np.sum(
                    protected_missing_mask
                )
            )

            # =================================================
            # GRAD UTILIZARE
            # =================================================

            if valid_pixels > 0:

                usability = (
                        feasible_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

                lc_excluded_pct = (
                        lc_excluded_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

                lc_missing_pct = (
                        lc_missing_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

                hard_landcover_rejected_pct = (
                        hard_landcover_rejected_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

                protected_pct = (
                        protected_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

                protected_missing_pct = (
                        protected_missing_pixels
                        /
                        valid_pixels
                        *
                        100.0
                )

            else:

                usability = 0.0
                lc_excluded_pct = 0.0
                lc_missing_pct = 0.0
                hard_landcover_rejected_pct = 0.0
                protected_pct = 0.0
                protected_missing_pct = 0.0

            # =================================================
            # SUPRAFAȚĂ FEZABILĂ
            # =================================================

            suitable_area_ha = (
                    total_area_ha
                    *
                    usability
                    /
                    100.0
            )

            # =================================================
            # SCOR PROCENTUAL
            # =================================================

            if score_mean is not None:

                score_percent = (
                        score_mean
                        *
                        100.0
                )

            else:

                score_percent = None

            # =================================================
            # CONSTRÂNGERI / REZUMAT PENTRU UI
            # =================================================

            # Procentul de teren fezabil este calculat strict din
            # pixelii valizi ai parcelei. Îl păstrăm la 2 zecimale
            # pentru afișare exactă în interfață.
            feasible_pct = float(usability)
            infeasible_pct = max(0.0, 100.0 - feasible_pct)

            constraints = []

            # -------------------------------------------------
            # CLASE LAND COVER HARD - GRUPATE PE CATEGORII
            # -------------------------------------------------

            forest_codes = set()
            sealed_codes = set()
            water_codes = set()
            snow_codes = set()

            if lc_mode == "CLCplus Backbone":
                forest_codes = {2, 3, 4}
                sealed_codes = {1}
                water_codes = {10}
                snow_codes = {11}

            elif lc_mode == "ESA WorldCover":
                forest_codes = {10}
                sealed_codes = {50}
                water_codes = {80, 90, 95}
                snow_codes = {70}

            elif lc_mode == "CORINE Land Cover":
                forest_codes = {311, 312, 313}
                sealed_codes = {
                    111, 112, 121, 122, 123, 124,
                    131, 132, 133, 141, 142
                }
                water_codes = {
                    411, 412, 421, 422, 423,
                    511, 512, 521, 522, 523
                }
                snow_codes = {335}

            forest_pct = sum(
                float(item.get("percentage", 0.0))
                for item in lc_class_summary
                if int(item.get("code", -999999)) in forest_codes
            )

            sealed_pct = sum(
                float(item.get("percentage", 0.0))
                for item in lc_class_summary
                if int(item.get("code", -999999)) in sealed_codes
            )

            water_pct = sum(
                float(item.get("percentage", 0.0))
                for item in lc_class_summary
                if int(item.get("code", -999999)) in water_codes
            )

            snow_pct = sum(
                float(item.get("percentage", 0.0))
                for item in lc_class_summary
                if int(item.get("code", -999999)) in snow_codes
            )

            # -------------------------------------------------
            # MESAJ PRINCIPAL - FĂRĂ REPETIȚII
            # -------------------------------------------------

            if protected_pct > 0.005:
                constraints.append(
                    "Constrângere majoră: Arie protejată Natura 2000 — "
                    f"{protected_pct:.2f}% din parcelă se suprapune cu "
                    "un sit Natura 2000 și este exclusă din analiza de "
                    "fezabilitate."
                )

            if protected_missing_pct > 0.005:
                constraints.append(
                    "Date Natura 2000 indisponibile pentru "
                    f"{protected_missing_pct:.2f}% din parcelă; această "
                    "zonă este tratată conservator ca nefezabilă."
                )

            if forest_pct > 0.005:
                constraints.append(
                    "Constrângere majoră: Pădure — "
                    f"{forest_pct:.2f}% din parcelă este acoperită "
                    "de vegetație arboricolă și este exclusă din "
                    "analiza de fezabilitate."
                )

            if sealed_pct > 0.005:
                constraints.append(
                    "Constrângere majoră: Suprafață construită / sigilată — "
                    f"{sealed_pct:.2f}% din parcelă este exclusă."
                )

            if water_pct > 0.005:
                constraints.append(
                    "Constrângere majoră: Apă / zonă umedă — "
                    f"{water_pct:.2f}% din parcelă este exclusă."
                )

            if snow_pct > 0.005:
                constraints.append(
                    "Constrângere majoră: Zăpadă / gheață — "
                    f"{snow_pct:.2f}% din parcelă este exclusă."
                )

            if lc_missing_pixels > 0:
                constraints.append(
                    "Date Land Cover indisponibile pentru "
                    f"{lc_missing_pct:.2f}% din parcelă; această zonă "
                    "este tratată conservator ca nefezabilă."
                )

            if (
                    slope_mean is not None
                    and
                    slope_mean > 15.0
            ):
                constraints.append(
                    f"Înclinație medie ridicată: {slope_mean:.2f}°."
                )

            if (
                    distance_mean is not None
                    and
                    distance_mean > 2000.0
            ):
                constraints.append(
                    f"Distanță mare față de rețea: {distance_mean:.0f} m."
                )

            if (
                    score_percent is not None
                    and
                    score_percent < SCORE_THRESHOLD * 100
            ):
                constraints.append(
                    "Scor MCDA mediu sub pragul minim de fezabilitate."
                )

            # Afișăm o singură dată procentul final, exact la 2 zecimale.
            constraints.append(
                f"Fezabilitate: {feasible_pct:.2f}% din parcelă "
                f"({suitable_area_ha:.2f} ha din {total_area_ha:.2f} ha)."
            )

            # =================================================
            # EVALUARE FINALĂ PENTRU UI
            # =================================================

            primary_reason = None

            if protected_pct >= 99.995:
                primary_reason = "suprapunere integrală cu arie protejată Natura 2000"
            elif forest_pct >= 99.995:
                primary_reason = "suprapunere integrală cu zonă forestieră"
            elif sealed_pct >= 99.995:
                primary_reason = "suprapunere integrală cu suprafață construită / sigilată"
            elif water_pct >= 99.995:
                primary_reason = "suprapunere integrală cu apă / zonă umedă"
            elif snow_pct >= 99.995:
                primary_reason = "suprapunere integrală cu zăpadă / gheață"
            elif protected_pct > 0.005:
                primary_reason = f"{protected_pct:.2f}% suprapunere cu arie protejată Natura 2000"
            elif forest_pct > 0.005:
                primary_reason = f"{forest_pct:.2f}% suprapunere cu zonă forestieră"
            elif sealed_pct > 0.005:
                primary_reason = f"{sealed_pct:.2f}% suprafață construită / sigilată"
            elif water_pct > 0.005:
                primary_reason = f"{water_pct:.2f}% apă / zonă umedă"

            if feasible_pct <= 0.005:
                final_evaluation_level = "bad"
                if primary_reason:
                    final_evaluation = (
                        "Teren nefezabil pentru parc solar. "
                        f"Motiv: {primary_reason}. "
                        "Fezabilitate: 0.00%."
                    )
                else:
                    final_evaluation = (
                        "Teren nefezabil pentru parc solar. "
                        "Fezabilitate: 0.00%."
                    )

            elif feasible_pct < 60.0:
                final_evaluation_level = "bad"
                final_evaluation = (
                    "Teren cu fezabilitate redusă: "
                    f"{feasible_pct:.2f}% din parcelă este fezabilă."
                )

            elif feasible_pct < 80.0:
                final_evaluation_level = "warning"
                final_evaluation = (
                    "Teren parțial fezabil pentru parc solar: "
                    f"{feasible_pct:.2f}% din parcelă este fezabilă."
                )

            else:
                final_evaluation_level = "good"
                final_evaluation = (
                    "Teren favorabil pentru parc solar: "
                    f"{feasible_pct:.2f}% din parcelă este fezabilă."
                )

            # =================================================
            # DEBUG
            # =================================================

            print()
            print(
                "Mod Land Cover:",
                lc_mode
            )

            print(
                "Pixeli valizi:",
                valid_pixels
            )

            print(
                "Pixeli excluși HARD Land Cover:",
                lc_excluded_pixels
            )

            print(
                "Pixeli fără Land Cover:",
                lc_missing_pixels
            )

            print(
                "Pixeli respinși total de Land Cover:",
                hard_landcover_rejected_pixels
            )

            print(
                "Pixeli Natura 2000:",
                protected_pixels
            )

            print(
                "Procent Natura 2000:",
                f"{protected_pct:.2f}%"
            )

            print(
                "Pixeli fără date Natura 2000:",
                protected_missing_pixels
            )

            print(
                "Clase Land Cover în parcelă:"
            )

            for item in lc_class_summary:

                print(
                    "  -",
                    item["code"],
                    item["name"],
                    f"{item['percentage']:.2f}%",
                    "INTERZIS"
                    if item["forbidden"]
                    else "permis"
                )

            print(
                "Pixeli fezabili:",
                feasible_pixels
            )

            print(
                "Grad utilizare:",
                round(
                    usability,
                    2
                ),
                "%"
            )

            print(
                "Suprafață fezabilă:",
                round(
                    suitable_area_ha,
                    4
                ),
                "ha"
            )

            print("-" * 72)
            print()

            # =================================================
            # JSON
            # =================================================

            return jsonify({

                "status": "success",

                "details": {

                    "total_area":
                        round(
                            total_area_ha,
                            2
                        ),

                    "suitable_area":
                        round(
                            suitable_area_ha,
                            2
                        ),

                    "usability":
                        round(
                            usability,
                            2
                        ),

                    "feasible_percentage":
                        round(
                            feasible_pct,
                            2
                        ),

                    "infeasible_percentage":
                        round(
                            infeasible_pct,
                            2
                        ),

                    "final_evaluation":
                        final_evaluation,

                    "final_evaluation_level":
                        final_evaluation_level,

                    "avg_score":
                        (
                            round(
                                score_percent,
                                1
                            )
                            if
                            score_percent
                            is not None
                            else None
                        ),

                    "inclinatie_grade":
                        (
                            round(
                                slope_mean,
                                1
                            )
                            if
                            slope_mean
                            is not None
                            else None
                        ),

                    "orientare_grade":
                        (
                            round(
                                aspect_mean,
                                1
                            )
                            if
                            aspect_mean
                            is not None
                            else None
                        ),

                    "distanta_retea_m":
                        (
                            round(
                                distance_mean,
                                0
                            )
                            if
                            distance_mean
                            is not None
                            else None
                        ),

                    "iradiere_kwh":
                        (
                            round(
                                irradiation_mean,
                                0
                            )
                            if
                            irradiation_mean
                            is not None
                            else None
                        ),

                    "constraints":
                        constraints,

                    "landcover": {

                        "source":
                            LANDCOVER_PATH,

                        "mode":
                            lc_mode,

                        "hard_excluded_pct":
                            round(
                                lc_excluded_pct,
                                2
                            ),

                        "missing_pct":
                            round(
                                lc_missing_pct,
                                2
                            ),

                        "rejected_total_pct":
                            round(
                                hard_landcover_rejected_pct,
                                2
                            ),

                        "classes":
                            [
                                {
                                    **item,
                                    "percentage":
                                        round(
                                            item["percentage"],
                                            2
                                        )
                                }
                                for item in lc_class_summary
                            ]
                    },

                    "protected_areas": {
                        "source":
                            PROTECTED_AREAS_PATH,

                        "type":
                            "Natura 2000",

                        "overlap_pct":
                            round(
                                protected_pct,
                                2
                            ),

                        "missing_pct":
                            round(
                                protected_missing_pct,
                                2
                            ),

                        "overlap_pixels":
                            protected_pixels
                    },

                    "debug": {

                        "landcover_mode":
                            lc_mode,

                        "landcover_path":
                            LANDCOVER_PATH,

                        "valid_pixels":
                            valid_pixels,

                        "feasible_pixels":
                            feasible_pixels,

                        "landcover_excluded_pixels":
                            lc_excluded_pixels,

                        "landcover_missing_pixels":
                            lc_missing_pixels,

                        "protected_areas_path":
                            PROTECTED_AREAS_PATH,

                        "protected_pixels":
                            protected_pixels,

                        "protected_missing_pixels":
                            protected_missing_pixels,

                        "score_threshold":
                            SCORE_THRESHOLD
                    }
                }
            })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ============================================================
# DETALII PUNCT
# ============================================================

@app.route(
    "/api/get_site_details",
    methods=["POST"]
)
def get_site_details():

    data = request.get_json(
        silent=True
    )

    if (
            not data
            or
            "lon" not in data
            or
            "lat" not in data
    ):

        return jsonify({
            "error":
                "Coordonatele lipsesc!"
        }), 400

    try:

        lon = float(
            data.get("lon")
        )

        lat = float(
            data.get("lat")
        )

        if not os.path.exists(
                LANDCOVER_PATH
        ):

            raise FileNotFoundError(
                "Raster Land Cover real lipsă: "
                f"{LANDCOVER_PATH}"
            )

        if not os.path.exists(
                PROTECTED_AREAS_PATH
        ):

            raise FileNotFoundError(
                "Raster Natura 2000 lipsă: "
                f"{PROTECTED_AREAS_PATH}"
            )

        with rasterio.open(
                RASTER_PATH
        ) as src:

            if src.count < 5:

                raise ValueError(
                    "Rasterul MCDA trebuie să "
                    "aibă minimum 5 benzi."
                )

            if src.crs is None:

                raise ValueError(
                    "Rasterul MCDA nu are CRS."
                )

            # =================================================
            # TRANSFORMARE COORDONATE PENTRU MCDA
            # =================================================

            if src.crs.to_epsg() != 4326:

                transformer = (
                    Transformer.from_crs(
                        "EPSG:4326",
                        src.crs,
                        always_xy=True
                    )
                )

                x, y = transformer.transform(
                    lon,
                    lat
                )

            else:

                x = lon
                y = lat

            # =================================================
            # CITIRE PIXEL MCDA
            # =================================================

            valori_pixel = np.asarray(
                list(
                    src.sample(
                        [(x, y)]
                    )
                )[0],
                dtype=np.float64
            )

            score = float(
                valori_pixel[0]
            )

            slope = float(
                valori_pixel[1]
            )

            aspect = float(
                valori_pixel[2]
            )

            distance = float(
                valori_pixel[3]
            )

            irradiation = float(
                valori_pixel[4]
            )

            # =================================================
            # LAND COVER DIN RASTERUL SEPARAT
            # =================================================

            landcover = sample_landcover_wgs84(
                lon,
                lat
            )

            protected_value = sample_binary_raster_wgs84(
                PROTECTED_AREAS_PATH,
                lon,
                lat
            )

            protected_missing = not np.isfinite(
                protected_value
            )

            is_protected = (
                    np.isfinite(protected_value)
                    and
                    protected_value >= 0.5
            )

            if (
                    not np.isfinite(score)
                    or
                    score < 0.0
                    or
                    score > 1.0
            ):

                return jsonify({

                    "status": "error",

                    "message":
                        "Punctul nu are "
                        "un scor valid."

                }), 400

            lc_array = np.array(
                [[landcover]],
                dtype=np.float64
            )

            valid_array = np.array(
                [[True]],
                dtype=bool
            )

            (
                lc_allowed,
                lc_mode,
                _,
                lc_missing,
                lc_class_summary

            ) = infer_landcover_allowed_mask(
                lc_array,
                valid_array,
                source_path=LANDCOVER_PATH,
                scheme=LANDCOVER_SCHEME
            )

            is_feasible = (
                    score
                    >=
                    SCORE_THRESHOLD
                    and
                    bool(
                        lc_allowed[0, 0]
                    )
                    and
                    (not is_protected)
                    and
                    (not protected_missing)
            )

            return jsonify({

                "status":
                    (
                        "Fezabil"
                        if is_feasible
                        else "Nefezabil"
                    ),

                "detalii": {

                    "scor":
                        round(
                            score * 100.0,
                            1
                        ),

                    "inclinatie_grade":
                        (
                            round(
                                slope,
                                1
                            )
                            if
                            np.isfinite(slope)
                            else None
                        ),

                    "orientare_grade":
                        (
                            round(
                                aspect % 360.0,
                                1
                            )
                            if
                            np.isfinite(aspect)
                            else None
                        ),

                    "distanta_retea_m":
                        (
                            round(
                                distance,
                                0
                            )
                            if
                            np.isfinite(distance)
                            else None
                        ),

                    "iradiere_kwh":
                        (
                            round(
                                irradiation,
                                0
                            )
                            if
                            np.isfinite(
                                irradiation
                            )
                            else None
                        ),

                    "landcover_mode":
                        lc_mode,

                    "landcover_source":
                        LANDCOVER_PATH,

                    "landcover_value":
                        (
                            round(
                                landcover,
                                3
                            )
                            if
                            np.isfinite(
                                landcover
                            )
                            else None
                        ),

                    "landcover_missing":
                        bool(
                            lc_missing > 0
                        ),

                    "natura2000_source":
                        PROTECTED_AREAS_PATH,

                    "natura2000_value":
                        (
                            round(
                                protected_value,
                                3
                            )
                            if
                            np.isfinite(protected_value)
                            else None
                        ),

                    "natura2000_protected":
                        bool(is_protected),

                    "natura2000_missing":
                        bool(protected_missing),

                    "landcover_class":
                        (
                            lc_class_summary[0]
                            if
                            lc_class_summary
                            else None
                        )
                }
            })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error":
                str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "raster":
            RASTER_PATH,

        "raster_exists":
            os.path.exists(
                RASTER_PATH
            ),

        "landcover":
            LANDCOVER_PATH,

        "landcover_exists":
            os.path.exists(
                LANDCOVER_PATH
            ),

        "landcover_scheme":
            LANDCOVER_SCHEME,

        "protected_areas":
            PROTECTED_AREAS_PATH,

        "protected_areas_exists":
            os.path.exists(
                PROTECTED_AREAS_PATH
            )
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print_raster_info()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
