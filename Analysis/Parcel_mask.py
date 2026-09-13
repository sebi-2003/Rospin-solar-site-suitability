import numpy as np
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


def create_parcel_mask(
        polygon_geojson,
        raster_crs,
        raster_transform,
        raster_shape,
        polygon_crs="EPSG:4326"
):
    """
    Creează o mască booleană pentru parcela desenată în Leaflet.

    True  = pixel aflat în interiorul parcelei
    False = pixel din afara parcelei

    Leaflet trimite în mod normal coordonate EPSG:4326.
    Rasterul poate fi, de exemplu, EPSG:3844.
    """

    if polygon_geojson is None:
        raise ValueError(
            "polygon_geojson este None."
        )

    if raster_crs is None:
        raise ValueError(
            "Rasterul nu are CRS."
        )

    # --------------------------------------------------------
    # Reproiectăm poligonul în CRS-ul rasterului
    # --------------------------------------------------------

    polygon_projected = transform_geom(
        polygon_crs,
        raster_crs,
        polygon_geojson,
        precision=6
    )

    # --------------------------------------------------------
    # Rasterizăm poligonul
    # --------------------------------------------------------

    parcel_mask = geometry_mask(
        [polygon_projected],
        out_shape=raster_shape,
        transform=raster_transform,
        invert=True,
        all_touched=False
    )

    parcel_pixels = np.count_nonzero(
        parcel_mask
    )

    if parcel_pixels == 0:
        raise ValueError(
            "\nEROARE: parcela nu intersectează rasterul.\n"
            "Verifică CRS-ul poligonului și al rasterului."
        )

    print("\n" + "=" * 70)
    print("PARCEL MASK")
    print("=" * 70)

    print(
        "Pixeli în interiorul parcelei:",
        parcel_pixels
    )

    print(
        "Pixeli raster total:",
        parcel_mask.size
    )

    print(
        f"Procent raster acoperit de parcelă: "
        f"{parcel_pixels / parcel_mask.size * 100:.4f}%"
    )

    return parcel_mask