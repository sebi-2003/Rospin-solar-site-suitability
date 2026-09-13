import numpy as np
from scipy.ndimage import distance_transform_edt


def check_buffer_zones(
        restricted_feature_data,
        pixel_resolution=10.0,
        buffer_distance_meters=50.0,
        feature_values=None,
        nodata_value=None,
        nodata_is_unsafe=True,
        feature_name="element restricționat"
):
    """
    Calculează bufferul față de un element restricționat.

    PARAMETRI
    ----------
    restricted_feature_data:
        Raster 2D.

    pixel_resolution:
        Poate fi:
            10.0
        sau:
            (10.0, 10.0)

        Pentru raster cu pixeli de dimensiuni diferite:
            (rezolutie_y, rezolutie_x)

    buffer_distance_meters:
        Distanța minimă în metri.

    feature_values:
        Valorile care reprezintă restricția.

        Exemple:
            [10]              -> pădure ESA WorldCover
            [311, 312, 313]   -> pădure CORINE
            [80]              -> apă WorldCover

        Dacă este None:
            boolean True = restricție
            sau orice valoare != 0 = restricție

    nodata_value:
        Valoarea NoData.

    nodata_is_unsafe:
        Dacă True, NoData va fi considerat nesigur.

    RETURN
    ------
    safe_mask:
        True  = respectă bufferul
        False = restricție / în interiorul bufferului
    """

    data = np.asarray(
        restricted_feature_data
    )

    if data.ndim != 2:
        raise ValueError(
            "restricted_feature_data trebuie "
            "să fie raster 2D."
        )

    print("\n" + "=" * 70)
    print(
        f"ANALIZĂ BUFFER: {feature_name}"
    )
    print("=" * 70)

    print(
        f"Buffer solicitat: "
        f"{buffer_distance_meters} m"
    )

    # ========================================================
    # REZOLUȚIA RASTERULUI
    # ========================================================

    if np.isscalar(pixel_resolution):

        resolution_y = abs(
            float(pixel_resolution)
        )

        resolution_x = abs(
            float(pixel_resolution)
        )

    else:

        if len(pixel_resolution) != 2:
            raise ValueError(
                "pixel_resolution trebuie să fie "
                "număr sau tuple (y, x)."
            )

        resolution_y = abs(
            float(pixel_resolution[0])
        )

        resolution_x = abs(
            float(pixel_resolution[1])
        )

    if resolution_x <= 0 or resolution_y <= 0:
        raise ValueError(
            "Rezoluția pixelului trebuie "
            "să fie > 0."
        )

    print(
        f"Rezoluție raster: "
        f"{resolution_x:.4f} × "
        f"{resolution_y:.4f} m"
    )

    # ========================================================
    # NODATA
    # ========================================================

    valid_mask = np.ones(
        data.shape,
        dtype=bool
    )

    if np.issubdtype(
            data.dtype,
            np.floating
    ):
        valid_mask &= np.isfinite(data)

    if nodata_value is not None:
        valid_mask &= (
                data != nodata_value
        )

    # ========================================================
    # DETECTARE ELEMENT RESTRICȚIONAT
    # ========================================================

    if data.dtype == bool:

        is_feature = data.copy()

    elif feature_values is not None:

        is_feature = np.isin(
            data,
            feature_values
        )

    else:

        is_feature = (
                valid_mask &
                (data != 0)
        )

    is_feature &= valid_mask

    feature_count = np.count_nonzero(
        is_feature
    )

    print(
        f"Pixeli '{feature_name}': "
        f"{feature_count}"
    )

    if feature_count == 0:

        unique_values = np.unique(
            data[valid_mask]
        )

        print("\n")
        print("!" * 70)
        print(
            "ATENȚIE: NU A FOST DETECTAT "
            "NICIUN ELEMENT RESTRICȚIONAT."
        )
        print(
            "Valorile rasterului sunt:"
        )
        print(
            unique_values[:100]
        )
        print("!" * 70)

        # Important:
        # nu mai returnăm automat "totul sigur"
        # fără să avertizăm clar.

        safe_mask = valid_mask.copy()

        if nodata_is_unsafe:
            safe_mask &= valid_mask

        return safe_mask

    # ========================================================
    # DISTANCE TRANSFORM
    # ========================================================

    # ~is_feature:
    # False la elementul restricționat
    # True în rest
    #
    # sampling transformă direct
    # distanța în metri.
    distance_in_meters = (
        distance_transform_edt(
            ~is_feature,
            sampling=(
                resolution_y,
                resolution_x
            )
        )
    )

    # ========================================================
    # BUFFER
    # ========================================================

    safe_mask = (
            distance_in_meters >=
            float(buffer_distance_meters)
    )

    # Elementul propriu-zis este întotdeauna exclus.
    safe_mask[is_feature] = False

    # NoData - varianta conservatoare
    if nodata_is_unsafe:
        safe_mask[~valid_mask] = False

    # ========================================================
    # STATISTICI
    # ========================================================

    unsafe_mask = ~safe_mask

    violation_count = np.count_nonzero(
        unsafe_mask
    )

    safe_count = np.count_nonzero(
        safe_mask
    )

    total_pixels = safe_mask.size

    violation_percentage = (
            violation_count /
            total_pixels *
            100.0
    )

    safe_percentage = (
            safe_count /
            total_pixels *
            100.0
    )

    print("\n")
    print("=== RAPORT BUFFER ===")

    print(
        f"Element analizat: "
        f"{feature_name}"
    )

    print(
        f"Buffer: "
        f"{buffer_distance_meters:.2f} m"
    )

    print(
        f"Pixeli element restricționat: "
        f"{feature_count}"
    )

    print(
        f"Pixeli blocați "
        f"(element + buffer): "
        f"{violation_count}"
    )

    print(
        f"Suprafață blocată: "
        f"{violation_percentage:.2f}%"
    )

    print(
        f"Suprafață sigură: "
        f"{safe_percentage:.2f}%"
    )

    if violation_count == 0:

        print(
            "[OK] Zona respectă complet "
            "distanța minimă."
        )

    else:

        print(
            "[ATENȚIE] O parte din teren "
            "se află în zona restricționată "
            "sau în buffer."
        )

    return safe_mask


# ============================================================
# HELPER PENTRU RASTERIO
# ============================================================

def get_pixel_resolution_from_transform(
        transform
):
    """
    Extrage automat rezoluția din transformarea
    unui raster Rasterio.

    Exemplu:
        with rasterio.open(path) as src:
            resolution = (
                get_pixel_resolution_from_transform(
                    src.transform
                )
            )
    """

    resolution_x = abs(
        float(transform.a)
    )

    resolution_y = abs(
        float(transform.e)
    )

    if resolution_x <= 0 or resolution_y <= 0:
        raise ValueError(
            "Transform raster invalid."
        )

    return (
        resolution_y,
        resolution_x
    )