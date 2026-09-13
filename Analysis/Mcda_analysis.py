import os
import numpy as np
import rasterio
import geopandas as gpd

from scipy.ndimage import distance_transform_edt
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ============================================================
# CONFIGURARE
# ============================================================

MAX_SLOPE_DEG = 15.0

# Distanța de la drum folosită pentru scor.
# ATENȚIE: acesta este DRUM, nu rețeaua electrică.
MAX_ROAD_DISTANCE_M = 2000.0


# ============================================================
# CLCPLUS BACKBONE 2023
# ============================================================
#
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
#
# 253 = Coastal seawater buffer
# 254 = Outside area
# 255 = NoData
#
# Pentru screening-ul tehnic al unui parc solar,
# considerăm inițial drept clase candidate:
#
# 6 = permanent herbaceous
# 7 = periodically herbaceous
# 9 = non/sparsely vegetated
#
# Aceste reguli NU reprezintă automat reguli juridice.
# ============================================================

CLCPLUS_CANDIDATE_CLASSES = {
    6,
    7,
    9
}

CLCPLUS_REAL_CLASSES = set(
    range(1, 12)
)

CLCPLUS_SPECIAL_CLASSES = {
    253,
    254,
    255
}


# ============================================================
# HELPER - VERIFICARE FIȘIER
# ============================================================

def check_file(path, name):
    if not path:
        return False

    if not os.path.exists(path):
        print(
            f"[WARNING] {name} nu există:"
        )
        print(
            f"          {path}"
        )
        return False

    return True


# ============================================================
# VERIFICARE LAND COVER
# ============================================================

def validate_landcover_file(
        landcover_path,
        land_cover
):
    """
    Verifică să NU fie folosit accidental CPMCE,
    CTYCL sau alt produs Copernicus care nu este
    o hartă generală de Land Cover.
    """

    filename = os.path.basename(
        landcover_path
    ).upper()

    # --------------------------------------------------------
    # EROAREA PE CARE O AVEAI
    # --------------------------------------------------------

    if "CPMCE" in filename:

        raise ValueError(
            "\n\n"
            "EROARE LAND COVER!\n"
            "Ai selectat un raster CPMCE "
            "(Main Crop Emergence).\n\n"
            "Acesta NU este un raster general "
            "Land Cover.\n"
            "Valoarea 0 în CPMCE înseamnă "
            "'No annual cropland'.\n\n"
            "Pentru această analiză folosește "
            "CLCplus Backbone RAS, de exemplu:\n\n"
            "CLMS_CLCPLUS_RAS_S2023_R10m_"
            "E53N27_03035_V01_R00.tif\n"
        )

    finite = np.asarray(
        land_cover
    )

    unique_values = np.unique(
        finite
    )

    print()
    print("=" * 70)
    print("LAND COVER - VALORI DETECTATE")
    print("=" * 70)

    print(
        unique_values[:100]
    )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Dacă numele spune explicit CLCPLUS,
    # facem și verificarea claselor.
    # --------------------------------------------------------

    if "CLCPLUS" in filename:

        integer_values = set(
            int(x)
            for x in unique_values
            if np.isfinite(x)
        )

        expected = (
                CLCPLUS_REAL_CLASSES
                |
                CLCPLUS_SPECIAL_CLASSES
        )

        unexpected = (
                integer_values
                -
                expected
        )

        if unexpected:

            raise ValueError(
                "Rasterul pare CLCplus, "
                "dar conține coduri neașteptate: "
                f"{sorted(unexpected)[:30]}"
            )


# ============================================================
# REPROIECTARE RASTER
# ============================================================

def align_raster_to_grid(
        raster_path,
        dst_shape,
        dst_transform,
        dst_crs,
        resampling=Resampling.bilinear
):
    """
    Reproiectează un raster pe grila principală.
    """

    destination = np.full(
        dst_shape,
        np.nan,
        dtype=np.float32
    )

    with rasterio.open(
            raster_path
    ) as src:

        reproject(
            source=rasterio.band(
                src,
                1
            ),

            destination=destination,

            src_transform=src.transform,
            src_crs=src.crs,

            src_nodata=src.nodata,

            dst_transform=dst_transform,
            dst_crs=dst_crs,

            dst_nodata=np.nan,

            resampling=resampling
        )

    return destination


# ============================================================
# VECTOR -> RASTER
# ============================================================

def vector_to_mask(
        vector_path,
        shape,
        transform,
        target_crs,
        layer=None,
        all_touched=True
):
    """
    Rasterizează un vector.

    Returnează:
        0 = în afara geometriei
        1 = în interior / intersectat
    """

    empty = np.zeros(
        shape,
        dtype=np.uint8
    )

    if not check_file(
            vector_path,
            "vector"
    ):
        return empty

    try:

        if layer is not None:

            try:
                gdf = gpd.read_file(
                    vector_path,
                    layer=layer
                )

            except Exception:

                print(
                    f"[WARNING] Layer '{layer}' "
                    f"nu a putut fi citit."
                )

                print(
                    "Se încearcă primul layer."
                )

                gdf = gpd.read_file(
                    vector_path
                )

        else:

            gdf = gpd.read_file(
                vector_path
            )

        if gdf.empty:

            print(
                f"[WARNING] Vector gol: "
                f"{vector_path}"
            )

            return empty

        if gdf.crs is None:

            raise ValueError(
                f"Vectorul {vector_path} "
                f"nu are CRS definit."
            )

        if gdf.crs != target_crs:

            gdf = gdf.to_crs(
                target_crs
            )

        gdf = gdf[
            gdf.geometry.notna()
        ].copy()

        gdf = gdf[
            ~gdf.geometry.is_empty
        ].copy()

        gdf = gdf[
            gdf.geometry.is_valid
        ].copy()

        if gdf.empty:

            return empty

        shapes = [
            (
                geom,
                1
            )
            for geom in gdf.geometry
        ]

        result = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
            all_touched=all_touched
        )

        return result

    except Exception as exc:

        print()
        print(
            "[ERROR] Nu am putut rasteriza:"
        )

        print(
            vector_path
        )

        print(
            repr(exc)
        )

        raise


# ============================================================
# PANTĂ + ASPECT
# ============================================================

def calculate_slope_and_aspect(
        dem,
        transform,
        crs
):
    """
    Calculează panta și aspectul.

    Grila principală CLCplus este EPSG:3035,
    deci rezoluția este exprimată în metri.
    """

    if crs is None:

        raise ValueError(
            "CRS lipsă."
        )

    if not crs.is_projected:

        raise ValueError(
            "Grila principală trebuie să fie "
            "într-un CRS proiectat metric. "
            "CLCplus Backbone folosește EPSG:3035."
        )

    res_x = abs(
        transform.a
    )

    res_y = abs(
        transform.e
    )

    print(
        f"Rezoluție analiză: "
        f"{res_x:.2f} x {res_y:.2f} m"
    )

    # --------------------------------------------------------
    # Gradientul array-ului
    #
    # row crește spre sud
    # col crește spre est
    # --------------------------------------------------------

    dz_drow, dz_dx = np.gradient(
        dem.astype(np.float64),
        res_y,
        res_x
    )

    # --------------------------------------------------------
    # PANTĂ
    # --------------------------------------------------------

    gradient = np.sqrt(
        dz_dx ** 2
        +
        dz_drow ** 2
    )

    slope = np.degrees(
        np.arctan(
            gradient
        )
    )

    # --------------------------------------------------------
    # ASPECT
    #
    # 0   = N
    # 90  = E
    # 180 = S
    # 270 = W
    # --------------------------------------------------------

    aspect = np.degrees(
        np.arctan2(
            -dz_dx,
            dz_drow
        )
    )

    aspect = (
                     aspect
                     +
                     360.0
             ) % 360.0

    # Pe teren aproape perfect plat,
    # aspectul nu are semnificație.
    aspect = np.where(
        slope < 0.5,
        np.nan,
        aspect
    )

    return (
        slope.astype(np.float32),
        aspect.astype(np.float32)
    )


# ============================================================
# DISTANȚA PÂNĂ LA DRUM
# ============================================================

def calculate_road_distance(
        roads_mask,
        transform
):
    """
    Calculează distanța până la cel mai apropiat drum.

    ATENȚIE:
    aceasta este distanța la DRUM,
    nu distanța la rețeaua electrică.
    """

    res_x = abs(
        transform.a
    )

    res_y = abs(
        transform.e
    )

    if not np.any(
            roads_mask == 1
    ):

        return np.full(
            roads_mask.shape,
            np.nan,
            dtype=np.float32
        )

    distances = distance_transform_edt(
        roads_mask == 0,
        sampling=(
            res_y,
            res_x
        )
    )

    return distances.astype(
        np.float32
    )


# ============================================================
# SCOR ASPECT
# ============================================================

def calculate_aspect_score(
        aspect,
        slope
):
    """
    Scor continuu:
        Sud = 1
        Est/Vest ≈ 0.5
        Nord = 0

    Pentru teren aproape plat:
        scor = 1
    """

    score = (
                    1.0
                    +
                    np.cos(
                        np.radians(
                            aspect - 180.0
                        )
                    )
            ) / 2.0

    score = np.where(
        slope < 2.0,
        1.0,
        score
    )

    score = np.where(
        np.isfinite(score),
        score,
        1.0
    )

    return np.clip(
        score,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# SCOR PANTĂ
# ============================================================

def calculate_slope_score(
        slope
):
    """
    0°  -> 1
    15° -> 0

    Peste 15° este oricum excludere hard.
    """

    score = (
            1.0
            -
            (
                    slope /
                    MAX_SLOPE_DEG
            )
    )

    return np.clip(
        score,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# SCOR DRUM
# ============================================================

def calculate_road_score(
        distances
):
    """
    Distanță mică la drum = scor mai bun.

    0 m    -> 1
    2000 m -> 0
    """

    score = (
            1.0
            -
            (
                    distances /
                    MAX_ROAD_DISTANCE_M
            )
    )

    return np.clip(
        score,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# FUNCȚIA PRINCIPALĂ
# ============================================================

def run_mcda_and_save(
        dem_path,
        roads_shp_path,
        landcover_path,
        protected_areas_path,
        output_path,
        solar_radiation_path=None
):

    print()
    print("=" * 80)
    print("HELIO SITE EXPLORER - MCDA")
    print("=" * 80)

    # ========================================================
    # VERIFICARE FIȘIERE
    # ========================================================

    if not check_file(
            landcover_path,
            "Land Cover"
    ):
        raise FileNotFoundError(
            landcover_path
        )

    if not check_file(
            dem_path,
            "DEM"
    ):
        raise FileNotFoundError(
            dem_path
        )

    # ========================================================
    # 1. LAND COVER = GRILA PRINCIPALĂ
    # ========================================================

    print()
    print(
        "1. Se citește CLCplus Backbone..."
    )

    with rasterio.open(
            landcover_path
    ) as src_lc:

        land_cover = src_lc.read(
            1
        )

        meta = src_lc.meta.copy()

        transform = (
            src_lc.transform
        )

        crs = (
            src_lc.crs
        )

        shape = (
            src_lc.height,
            src_lc.width
        )

        lc_nodata = (
            src_lc.nodata
        )

    validate_landcover_file(
        landcover_path,
        land_cover
    )

    if crs is None:

        raise ValueError(
            "Rasterul Land Cover nu are CRS."
        )

    print(
        "CRS principal:",
        crs
    )

    print(
        "Shape principal:",
        shape
    )

    print(
        "Transform:",
        transform
    )

    # ========================================================
    # CONVERTIM CLASELE LA INTEGER
    # ========================================================

    land_cover_float = (
        land_cover.astype(
            np.float64
        )
    )

    land_cover_int = np.rint(
        land_cover_float
    ).astype(
        np.int32
    )

    # Numai clasele reale 1..11 sunt
    # considerate zonă geografică validă.
    valid_landcover = np.isin(
        land_cover_int,
        list(
            CLCPLUS_REAL_CLASSES
        )
    )

    if lc_nodata is not None:

        valid_landcover &= (
                land_cover_float
                !=
                lc_nodata
        )

    # ========================================================
    # STATISTICI LAND COVER
    # ========================================================

    print()
    print(
        "Clase CLCplus în tile:"
    )

    values, counts = np.unique(
        land_cover_int[
            valid_landcover
        ],
        return_counts=True
    )

    for value, count in zip(
            values,
            counts
    ):

        print(
            f"  clasa {value}: "
            f"{count} pixeli"
        )

    # ========================================================
    # 2. DEM -> GRILĂ CLCPLUS 10 M
    # ========================================================

    print()
    print(
        "2. Se aliniază DEM-ul "
        "la grila CLCplus..."
    )

    dem = align_raster_to_grid(
        dem_path,
        shape,
        transform,
        crs,
        resampling=Resampling.bilinear
    )

    dem_valid = np.isfinite(
        dem
    )

    # ========================================================
    # 3. PANTĂ + ASPECT
    # ========================================================

    print()
    print(
        "3. Se calculează panta "
        "și orientarea..."
    )

    slope_data, aspect_data = (
        calculate_slope_and_aspect(
            dem,
            transform,
            crs
        )
    )

    # ========================================================
    # 4. DRUMURI
    # ========================================================

    print()
    print(
        "4. Se rasterizează drumurile..."
    )

    roads_mask = vector_to_mask(
        roads_shp_path,
        shape,
        transform,
        crs
    )

    distances_data = (
        calculate_road_distance(
            roads_mask,
            transform
        )
    )

    # ========================================================
    # 5. NATURA 2000
    # ========================================================

    print()
    print(
        "5. Se aplică Natura 2000..."
    )

    natura_mask = vector_to_mask(
        protected_areas_path,
        shape,
        transform,
        crs,
        layer="NaturaSite_polygon"
    )

    # natura_mask:
    # 0 = în afara Natura 2000
    # 1 = în Natura 2000

    # ========================================================
    # 6. LAND COVER EXCLUSION
    # ========================================================

    print()
    print(
        "6. Se creează masca "
        "Land Cover..."
    )

    candidate_landcover = np.isin(
        land_cover_int,
        list(
            CLCPLUS_CANDIDATE_CLASSES
        )
    )

    landcover_exclusion = (
            valid_landcover
            &
            ~candidate_landcover
    )

    # ========================================================
    # 7. SLOPE EXCLUSION
    # ========================================================

    slope_exclusion = (
            np.isfinite(
                slope_data
            )
            &
            (
                    slope_data
                    >
                    MAX_SLOPE_DEG
            )
    )

    # ========================================================
    # 8. ROAD EXCLUSION
    # ========================================================
    #
    # Pixelul pe care se află efectiv drumul
    # nu poate fi considerat suprafață
    # disponibilă pentru panouri.
    # ========================================================

    road_exclusion = (
            roads_mask == 1
    )

    # ========================================================
    # 9. COMBINARE HARD EXCLUSIONS
    # ========================================================

    analysis_valid = (
            valid_landcover
            &
            dem_valid
            &
            np.isfinite(
                slope_data
            )
    )

    exclusion_mask = (
            landcover_exclusion
            |
            (natura_mask == 1)
            |
            slope_exclusion
            |
            road_exclusion
    )

    # În afara zonei valide nu calculăm.
    exclusion_mask &= (
        analysis_valid
    )

    # ========================================================
    # 10. SCORURI CONTINUE
    # ========================================================

    print()
    print(
        "7. Se calculează scorul MCDA..."
    )

    slope_score = (
        calculate_slope_score(
            slope_data
        )
    )

    aspect_score = (
        calculate_aspect_score(
            aspect_data,
            slope_data
        )
    )

    road_score = (
        calculate_road_score(
            distances_data
        )
    )

    # ========================================================
    # MCDA
    #
    # 45% pantă
    # 20% orientare
    # 35% acces la drum
    # ========================================================

    if np.any(
            np.isfinite(
                distances_data
            )
    ):

        score = (
                slope_score * 0.45
                +
                aspect_score * 0.20
                +
                road_score * 0.35
        )

    else:

        # Dacă nu avem drumuri,
        # nu inventăm o distanță de 500 m.
        #
        # Renormalizăm doar cele două criterii
        # disponibile.
        score = (
                slope_score * (
                0.45 / 0.65
        )
                +
                aspect_score * (
                        0.20 / 0.65
                )
        )

    score = score.astype(
        np.float32
    )

    # ========================================================
    # HARD EXCLUSION = SCOR 0
    #
    # Foarte important:
    # NU punem NaN pentru zonele excluse.
    #
    # Dacă le-am pune NaN, app.py le-ar elimina
    # din numitor și ai putea primi iar 100%
    # utilizare.
    # ========================================================

    score = np.where(
        analysis_valid,
        score,
        np.nan
    )

    score = np.where(
        exclusion_mask,
        0.0,
        score
    ).astype(
        np.float32
    )

    # ========================================================
    # 11. IRADIERE REALĂ
    # ========================================================

    print()
    print(
        "8. Se citește iradierea..."
    )

    if (
            solar_radiation_path
            and
            os.path.exists(
                solar_radiation_path
            )
    ):

        irradiation_data = (
            align_raster_to_grid(
                solar_radiation_path,
                shape,
                transform,
                crs,
                resampling=Resampling.bilinear
            )
        )

        finite_irr = (
            irradiation_data[
                np.isfinite(
                    irradiation_data
                )
            ]
        )

        if finite_irr.size:

            print(
                "Iradiere:",
                "min=",
                float(
                    np.min(
                        finite_irr
                    )
                ),
                "max=",
                float(
                    np.max(
                        finite_irr
                    )
                ),
                "mean=",
                float(
                    np.mean(
                        finite_irr
                    )
                )
            )

    else:

        # NU MAI INVENTĂM 1350
        irradiation_data = np.full(
            shape,
            np.nan,
            dtype=np.float32
        )

        print(
            "[WARNING] Nu există un raster "
            "de iradiere real."
        )

        print(
            "[WARNING] Banda 5 va fi NoData, "
            "nu 1350 inventat."
        )

    # ========================================================
    # 12. BANDA 6
    #
    # ACUM ESTE MASCA REALĂ DE EXCLUDERE:
    #
    # 0 = PERMIS
    # 1 = EXCLUS
    #
    # app.py pe care îl ai acum interpretează
    # exact această convenție.
    # ========================================================

    exclusion_band = np.where(
        analysis_valid,
        exclusion_mask.astype(
            np.float32
        ),
        np.nan
    ).astype(
        np.float32
    )

    # ========================================================
    # DEBUG
    # ========================================================

    total_valid = int(
        np.sum(
            analysis_valid
        )
    )

    excluded_lc = int(
        np.sum(
            landcover_exclusion
            &
            analysis_valid
        )
    )

    excluded_natura = int(
        np.sum(
            (natura_mask == 1)
            &
            analysis_valid
        )
    )

    excluded_slope = int(
        np.sum(
            slope_exclusion
            &
            analysis_valid
        )
    )

    excluded_roads = int(
        np.sum(
            road_exclusion
            &
            analysis_valid
        )
    )

    excluded_total = int(
        np.sum(
            exclusion_mask
        )
    )

    allowed_total = (
            total_valid
            -
            excluded_total
    )

    print()
    print("=" * 70)
    print("REZULTAT MĂȘTI")
    print("=" * 70)

    print(
        "Pixeli valizi:",
        total_valid
    )

    print(
        "Excluși Land Cover:",
        excluded_lc
    )

    print(
        "Excluși Natura 2000:",
        excluded_natura
    )

    print(
        "Excluși pantă > 15°:",
        excluded_slope
    )

    print(
        "Excluși drum:",
        excluded_roads
    )

    print(
        "TOTAL excluși:",
        excluded_total
    )

    print(
        "TOTAL candidați:",
        allowed_total
    )

    if total_valid > 0:

        print(
            "Procent candidat:",
            round(
                allowed_total
                /
                total_valid
                *
                100.0,
                2
            ),
            "%"
        )

    print("=" * 70)

    # ========================================================
    # 13. SALVARE
    # ========================================================

    meta.update(
        count=6,
        dtype=rasterio.float32,
        nodata=np.nan,
        compress="deflate"
    )

    output_dir = os.path.dirname(
        os.path.abspath(
            output_path
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print(
        "Se salvează:"
    )

    print(
        output_path
    )

    with rasterio.open(
            output_path,
            "w",
            **meta
    ) as dst:

        # 1
        dst.write(
            score.astype(
                np.float32
            ),
            1
        )

        # 2
        dst.write(
            slope_data.astype(
                np.float32
            ),
            2
        )

        # 3
        dst.write(
            aspect_data.astype(
                np.float32
            ),
            3
        )

        # 4
        dst.write(
            distances_data.astype(
                np.float32
            ),
            4
        )

        # 5
        dst.write(
            irradiation_data.astype(
                np.float32
            ),
            5
        )

        # 6
        dst.write(
            exclusion_band.astype(
                np.float32
            ),
            6
        )

        # ----------------------------------------------------
        # Descrieri benzi
        # ----------------------------------------------------

        dst.set_band_description(
            1,
            "MCDA score 0-1"
        )

        dst.set_band_description(
            2,
            "Slope degrees"
        )

        dst.set_band_description(
            3,
            "Aspect degrees 0=N 90=E 180=S 270=W"
        )

        dst.set_band_description(
            4,
            "Distance to road metres"
        )

        dst.set_band_description(
            5,
            "Solar irradiation"
        )

        dst.set_band_description(
            6,
            "Hard exclusion mask 0=allowed 1=excluded"
        )

    print()
    print("=" * 80)
    print(
        "MCDA FINALIZAT CU SUCCES"
    )
    print("=" * 80)
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_mcda_and_save(

        # ----------------------------------------------------
        # DEM
        # ----------------------------------------------------

        dem_path=(
            "../DEM/"
            "2026-08-25-00_00_2026-08-25-23_59_"
            "DEM_COPERNICUS_90_DEM_(Raw).tiff"
        ),

        # ----------------------------------------------------
        # DRUMURI
        # ----------------------------------------------------

        roads_shp_path=(
            "../Infrastructura/"
            "gis_osm_roads_free_1.shp"
        ),

        # ----------------------------------------------------
        # LAND COVER CORECT
        #
        # NU CPMCE!
        # ----------------------------------------------------

        landcover_path=(
            "../LandCover/"
            "CLMS_CLCPLUS_RAS_S2023_R10m_"
            "E53N27_03035_V01_R00.tiff"
        ),

        # ----------------------------------------------------
        # NATURA 2000
        # ----------------------------------------------------

        protected_areas_path=(
            "../AriiProtejate/"
            "Natura2000_end2024.gpkg"
        ),

        # ----------------------------------------------------
        # RASTER IRADIERE
        #
        # Dacă solar_final_clear.tif chiar conține
        # kWh/m²/an, îl folosim.
        # ----------------------------------------------------

        solar_radiation_path=(
            "../Analysis/"
            "solar_final_clear.tif"
        ),

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_path=(
            "../Backend/"
            "rezultat_suitabilitate.tif"
        )
    )