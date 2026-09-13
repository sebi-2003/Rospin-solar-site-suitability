import numpy as np


# ============================================================
# DEBUG IMPORT
# ============================================================

print("\n" + "=" * 70)
print(">>> Analysis/Exclusion_masks.py A FOST IMPORTAT <<<")
print("=" * 70 + "\n")


# ============================================================
# ESA WORLDCOVER 10 m
# ============================================================

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

WORLDCOVER_FORBIDDEN = [
    10,   # pădure
    50,   # construit
    70,   # zăpadă / gheață
    80,   # apă
    90,   # zonă umedă
    95,   # mangrove
]


# ============================================================
# CORINE LAND COVER
# ============================================================

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


CORINE_FORBIDDEN = [
    # urban / construit
    111,
    112,
    121,
    122,
    123,
    124,
    141,
    142,

    # pădure
    311,
    312,
    313,

    # zăpadă
    335,

    # wetlands
    411,
    412,
    421,
    422,
    423,

    # ape
    511,
    512,
    521,
    522,
    523,
]


# ============================================================
# DEBUG VALORI RASTER
# ============================================================

def print_unique_values(
        data,
        name,
        max_values=100
):
    """
    Afișează valorile unice și numărul de pixeli.
    """

    print("\n" + "-" * 60)
    print(f"DEBUG RASTER: {name}")
    print("-" * 60)

    if data is None:
        print(f"{name} = None")
        return

    data = np.asarray(data)

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    try:

        values, counts = np.unique(
            data,
            return_counts=True
        )

        print(
            f"Număr valori unice: "
            f"{len(values)}"
        )

        print("Valori:")

        for value, count in zip(
                values[:max_values],
                counts[:max_values]
        ):

            print(
                f"    {value} -> "
                f"{count} pixeli"
            )

        if len(values) > max_values:

            print(
                f"    ... încă "
                f"{len(values) - max_values} valori"
            )

    except Exception as e:

        print(
            "EROARE la np.unique:",
            e
        )


# ============================================================
# VALIDARE SHAPE
# ============================================================

def validate_shape(
        reference_data,
        other_data,
        other_name
):
    """
    Verifică dacă rasterele au aceeași dimensiune.
    """

    if other_data is None:
        return

    reference_data = np.asarray(
        reference_data
    )

    other_data = np.asarray(
        other_data
    )

    if (
            reference_data.shape !=
            other_data.shape
    ):

        raise ValueError(
            "\nEROARE DE ALINIERE!\n"
            f"Raster referință: "
            f"{reference_data.shape}\n"
            f"{other_name}: "
            f"{other_data.shape}\n\n"
            "Rasterele și parcel_mask trebuie "
            "să fie pe aceeași grilă."
        )


# ============================================================
# VALID MASK
# ============================================================

def _build_valid_mask(
        data,
        nodata_values=None
):
    """
    True = pixel valid.
    False = NaN / Inf / NoData.
    """

    data = np.asarray(
        data
    )

    valid_mask = np.ones(
        data.shape,
        dtype=bool
    )

    if np.issubdtype(
            data.dtype,
            np.floating
    ):

        valid_mask &= np.isfinite(
            data
        )

    if nodata_values is not None:

        for nodata in np.atleast_1d(
                nodata_values
        ):

            valid_mask &= (
                    data != nodata
            )

    return valid_mask


# ============================================================
# CONVERSIE LAND COVER LA INT
# ============================================================

def _integer_land_cover(
        data,
        valid_mask
):
    """
    Transformă valorile Land Cover în coduri întregi.

    Dacă există valori zecimale reale,
    înseamnă probabil că rasterul a fost
    reproiectat greșit cu Bilinear / Cubic.
    """

    data = np.asarray(
        data
    )

    if np.issubdtype(
            data.dtype,
            np.floating
    ):

        valid_values = data[
            valid_mask
        ]

        if valid_values.size > 0:

            difference = np.abs(
                valid_values -
                np.round(
                    valid_values
                )
            )

            if np.any(
                    difference > 1e-6
            ):

                raise ValueError(
                    "\nEROARE LAND COVER!\n"
                    "Rasterul conține valori zecimale.\n\n"
                    "Land Cover este categorial și "
                    "NU trebuie reproiectat cu "
                    "Bilinear/Cubic.\n\n"
                    "Folosește:\n"
                    "Resampling.nearest"
                )

    result = np.full(
        data.shape,
        -999999,
        dtype=np.int64
    )

    if np.any(
            valid_mask
    ):

        result[
            valid_mask
        ] = np.round(
            data[
                valid_mask
            ]
        ).astype(
            np.int64
        )

    return result


# ============================================================
# DETECTARE AUTOMATĂ LAND COVER
# ============================================================

def detect_land_cover_scheme(
        land_cover_data,
        nodata_values=(0, 255)
):
    """
    Detectează automat:
    - ESA WorldCover
    - CORINE Land Cover
    """

    data = np.asarray(
        land_cover_data
    )

    if data.ndim != 2:

        raise ValueError(
            "land_cover_data trebuie să fie "
            "un raster 2D."
        )

    valid_mask = _build_valid_mask(
        data,
        nodata_values=nodata_values
    )

    if not np.any(
            valid_mask
    ):

        raise ValueError(
            "Rasterul Land Cover nu conține "
            "pixeli valizi."
        )

    integer_data = _integer_land_cover(
        data,
        valid_mask
    )

    unique_values = set(
        np.unique(
            integer_data[
                valid_mask
            ]
        ).tolist()
    )

    print(
        "\nClase Land Cover detectate:",
        sorted(
            unique_values
        )
    )

    worldcover_codes = set(
        WORLDCOVER_CLASSES.keys()
    )

    corine_codes = set(
        CORINE_CLASSES.keys()
    )

    # --------------------------------------------------------
    # WORLDCOVER
    # --------------------------------------------------------

    if unique_values.issubset(
            worldcover_codes
    ):

        print(
            "[OK] Raster detectat: "
            "ESA WORLDCOVER"
        )

        return "worldcover"

    # --------------------------------------------------------
    # CORINE
    # --------------------------------------------------------

    if unique_values.issubset(
            corine_codes
    ):

        print(
            "[OK] Raster detectat: "
            "CORINE LAND COVER"
        )

        return "corine"

    # --------------------------------------------------------
    # NECUNOSCUT
    # --------------------------------------------------------

    print(
        "Clase WorldCover recunoscute:",
        sorted(
            unique_values &
            worldcover_codes
        )
    )

    print(
        "Clase CORINE recunoscute:",
        sorted(
            unique_values &
            corine_codes
        )
    )

    unknown_worldcover = sorted(
        unique_values -
        worldcover_codes
    )

    unknown_corine = sorted(
        unique_values -
        corine_codes
    )

    raise ValueError(
        "\nNU POT IDENTIFICA SIGUR "
        "LEGENDA LAND COVER.\n\n"
        f"Valorile găsite:\n"
        f"{sorted(unique_values)}\n\n"
        "Necunoscute pentru WorldCover:\n"
        f"{unknown_worldcover}\n\n"
        "Necunoscute pentru CORINE:\n"
        f"{unknown_corine}\n\n"
        "Analiza a fost oprită pentru "
        "a evita rezultate false."
    )


# ============================================================
# MASCĂ BINARĂ
# ============================================================

def create_binary_exclusion_mask(
        data,
        name,
        nodata_value=255,
        exclude_nodata=False
):
    """
    Pentru rastere binare.

    0 = liber
    orice valoare nenulă = restricție

    Return:
        True  = exclus
        False = permis
    """

    data = np.asarray(
        data
    )

    valid_mask = np.ones(
        data.shape,
        dtype=bool
    )

    if np.issubdtype(
            data.dtype,
            np.floating
    ):

        valid_mask &= np.isfinite(
            data
        )

    if nodata_value is not None:

        valid_mask &= (
                data != nodata_value
        )

    feature_mask = (
            valid_mask &
            (data != 0)
    )

    if exclude_nodata:

        exclusion_mask = (
                feature_mask |
                (~valid_mask)
        )

    else:

        exclusion_mask = (
            feature_mask
        )

    print_unique_values(
        data,
        name
    )

    print(
        f"\n{name}: "
        f"{np.count_nonzero(feature_mask)} "
        "pixeli cu restricție."
    )

    print(
        f"{name}: "
        f"{np.count_nonzero(~valid_mask)} "
        "pixeli NoData."
    )

    return exclusion_mask


# ============================================================
# LAND COVER EXCLUSION MASK
# ============================================================

def create_land_cover_exclusion_mask(
        land_cover_data,
        scheme="auto",
        forbidden_classes=None,
        nodata_values=(0, 255)
):
    """
    Creează masca de excludere Land Cover.

    RETURN:
        exclusion_mask
        detected_scheme
        used_forbidden_classes

    True  = exclus
    False = permis
    """

    data = np.asarray(
        land_cover_data
    )

    if data.ndim != 2:

        raise ValueError(
            "land_cover_data trebuie "
            "să fie raster 2D."
        )

    if scheme is None:

        scheme = "auto"

    scheme = str(
        scheme
    ).lower().strip()

    # ========================================================
    # DETECTARE
    # ========================================================

    if scheme == "auto":

        scheme = detect_land_cover_scheme(
            data,
            nodata_values=nodata_values
        )

    # ========================================================
    # WORLDCOVER
    # ========================================================

    if scheme == "worldcover":

        known_classes = set(
            WORLDCOVER_CLASSES.keys()
        )

        if forbidden_classes is None:

            forbidden_classes = (
                WORLDCOVER_FORBIDDEN
            )

    # ========================================================
    # CORINE
    # ========================================================

    elif scheme == "corine":

        known_classes = set(
            CORINE_CLASSES.keys()
        )

        if forbidden_classes is None:

            forbidden_classes = (
                CORINE_FORBIDDEN
            )

    else:

        raise ValueError(
            "land_cover_scheme trebuie "
            "să fie:\n"
            "'auto', 'worldcover' sau 'corine'."
        )

    forbidden_classes = [
        int(value)
        for value in forbidden_classes
    ]

    # ========================================================
    # VALID PIXELS
    # ========================================================

    valid_mask = _build_valid_mask(
        data,
        nodata_values=nodata_values
    )

    integer_data = _integer_land_cover(
        data,
        valid_mask
    )

    # ========================================================
    # CLASE NECUNOSCUTE
    # ========================================================

    known_mask = np.zeros(
        data.shape,
        dtype=bool
    )

    known_mask[
        valid_mask
    ] = np.isin(
        integer_data[
            valid_mask
        ],
        list(
            known_classes
        )
    )

    unknown_mask = (
            valid_mask &
            (~known_mask)
    )

    if np.any(
            unknown_mask
    ):

        unknown_values = np.unique(
            integer_data[
                unknown_mask
            ]
        )

        raise ValueError(
            "\nEROARE LAND COVER!\n"
            "Au fost găsite clase care "
            f"nu există în legenda "
            f"{scheme.upper()}:\n"
            f"{unknown_values.tolist()}\n\n"
            "Analiza a fost oprită."
        )

    # ========================================================
    # CLASE INTERZISE
    # ========================================================

    forbidden_mask = np.zeros(
        data.shape,
        dtype=bool
    )

    forbidden_mask[
        valid_mask
    ] = np.isin(
        integer_data[
            valid_mask
        ],
        forbidden_classes
    )

    # NoData = exclus conservator
    nodata_mask = (
        ~valid_mask
    )

    exclusion_mask = (
            forbidden_mask |
            nodata_mask
    )

    print("\n")
    print("=" * 60)
    print("LAND COVER EXCLUSIONS")
    print("=" * 60)

    print(
        "Legendă:",
        scheme.upper()
    )

    print(
        "Clase interzise:",
        forbidden_classes
    )

    print(
        "Pixeli Land Cover interziși:",
        np.count_nonzero(
            forbidden_mask
        )
    )

    print(
        "Pixeli Land Cover NoData:",
        np.count_nonzero(
            nodata_mask
        )
    )

    return (
        exclusion_mask,
        scheme,
        forbidden_classes
    )


# ============================================================
# FUNCȚIA PRINCIPALĂ
# ============================================================

def filter_exclusion_zones(
        land_cover_data,
        parcel_mask,
        protected_areas_data=None,
        urban_data=None,
        forbidden_classes=None,
        protected_nodata=255,
        urban_nodata=255,
        land_cover_scheme="auto",
        land_cover_nodata_values=(0, 255)
):
    """
    Creează masca FINALĂ de fezabilitate
    STRICT pentru parcela desenată.

    IMPORTANT:
        parcel_mask este obligatoriu.

    RETURN:
        True  = pixel fezabil din parcelă
        False = pixel exclus sau pixel din
                afara parcelei
    """

    print("\n\n")
    print("#" * 70)
    print(
        ">>> ANALIZĂ FEZABILITATE "
        "STRICT PE PARCELĂ <<<"
    )
    print("#" * 70)

    # ========================================================
    # 1. VALIDARE INPUT
    # ========================================================

    if land_cover_data is None:

        raise ValueError(
            "land_cover_data este None."
        )

    if parcel_mask is None:

        raise ValueError(
            "\nparcel_mask este None.\n\n"
            "Analiza NU va continua pe "
            "întreg rasterul.\n"
            "Trebuie creată mai întâi "
            "masca poligonului desenat."
        )

    land_cover_data = np.asarray(
        land_cover_data
    )

    parcel_mask = np.asarray(
        parcel_mask,
        dtype=bool
    )

    if land_cover_data.ndim != 2:

        raise ValueError(
            "land_cover_data trebuie "
            "să fie raster 2D.\n"
            f"Shape primit: "
            f"{land_cover_data.shape}"
        )

    if parcel_mask.ndim != 2:

        raise ValueError(
            "parcel_mask trebuie să "
            "fie mască 2D."
        )

    validate_shape(
        land_cover_data,
        parcel_mask,
        "parcel_mask"
    )

    # ========================================================
    # PROTECTED
    # ========================================================

    if protected_areas_data is not None:

        protected_areas_data = np.asarray(
            protected_areas_data
        )

        validate_shape(
            land_cover_data,
            protected_areas_data,
            "protected_areas_data"
        )

    # ========================================================
    # URBAN
    # ========================================================

    if urban_data is not None:

        urban_data = np.asarray(
            urban_data
        )

        validate_shape(
            land_cover_data,
            urban_data,
            "urban_data"
        )

    # ========================================================
    # 2. PIXELII PARCELEI
    # ========================================================

    total_parcel_pixels = (
        np.count_nonzero(
            parcel_mask
        )
    )

    if total_parcel_pixels == 0:

        raise ValueError(
            "\nEROARE:\n"
            "parcel_mask nu conține "
            "niciun pixel.\n\n"
            "Verifică CRS-ul poligonului "
            "și transformarea rasterului."
        )

    print(
        "\nPixeli în interiorul parcelei:",
        total_parcel_pixels
    )

    print(
        "Pixeli raster total:",
        parcel_mask.size
    )

    print(
        f"Parcela ocupă "
        f"{total_parcel_pixels / parcel_mask.size * 100:.4f}% "
        "din raster."
    )

    # ========================================================
    # 3. LAND COVER
    # ========================================================

    print_unique_values(
        land_cover_data,
        "LAND COVER"
    )

    (
        land_cover_exclusion,
        detected_scheme,
        used_forbidden_classes
    ) = create_land_cover_exclusion_mask(
        land_cover_data,
        scheme=land_cover_scheme,
        forbidden_classes=forbidden_classes,
        nodata_values=land_cover_nodata_values
    )

    # ========================================================
    # IMPORTANT:
    # Pornim STRICT de la parcelă.
    # ========================================================

    allowed_mask = (
        parcel_mask.copy()
    )

    # eliminăm restricțiile Land Cover
    allowed_mask &= (
        ~land_cover_exclusion
    )

    # ========================================================
    # 4. DISTRIBUȚIE LAND COVER ÎN PARCELĂ
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "LAND COVER ÎN INTERIORUL PARCELEI"
    )
    print("=" * 70)

    lc_valid_mask = _build_valid_mask(
        land_cover_data,
        nodata_values=land_cover_nodata_values
    )

    lc_integer = _integer_land_cover(
        land_cover_data,
        lc_valid_mask
    )

    valid_inside_parcel = (
            parcel_mask &
            lc_valid_mask
    )

    nodata_inside_parcel = (
            parcel_mask &
            (~lc_valid_mask)
    )

    print(
        "Pixeli Land Cover valizi "
        "în parcelă:",
        np.count_nonzero(
            valid_inside_parcel
        )
    )

    print(
        "Pixeli Land Cover NoData "
        "în parcelă:",
        np.count_nonzero(
            nodata_inside_parcel
        )
    )

    # ========================================================
    # CLASE ÎN PARCELĂ
    # ========================================================

    if np.any(
            valid_inside_parcel
    ):

        classes, counts = np.unique(
            lc_integer[
                valid_inside_parcel
            ],
            return_counts=True
        )

        for cls, count in zip(
                classes,
                counts
        ):

            cls_int = int(
                cls
            )

            percentage = (
                    count /
                    total_parcel_pixels *
                    100.0
            )

            if (
                    detected_scheme ==
                    "worldcover"
            ):

                name = (
                    WORLDCOVER_CLASSES.get(
                        cls_int,
                        "Necunoscut"
                    )
                )

            else:

                name = (
                    CORINE_CLASSES.get(
                        cls_int,
                        "Necunoscut"
                    )
                )

            if (
                    cls_int in
                    used_forbidden_classes
            ):

                status = "INTERZIS"

            else:

                status = "PERMIS"

            print(
                f"{cls_int:>3} | "
                f"{name:<42} | "
                f"{count:>8} px | "
                f"{percentage:>6.2f}% | "
                f"{status}"
            )

    # ========================================================
    # 5. ARII PROTEJATE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("ARII PROTEJATE")
    print("=" * 60)

    if protected_areas_data is not None:

        protected_mask = (
            create_binary_exclusion_mask(
                protected_areas_data,
                "PROTECTED AREAS",
                nodata_value=protected_nodata,
                exclude_nodata=False
            )
        )

        protected_inside = (
                parcel_mask &
                protected_mask
        )

        print(
            "Pixeli arii protejate "
            "în parcelă:",
            np.count_nonzero(
                protected_inside
            )
        )

        allowed_mask &= (
            ~protected_mask
        )

    else:

        print(
            "[AVERTISMENT] "
            "protected_areas_data = None."
        )

        print(
            "Ariile protejate NU sunt "
            "analizate."
        )

    # ========================================================
    # 6. URBAN
    # ========================================================

    print("\n")
    print("=" * 60)
    print("ZONE URBANE")
    print("=" * 60)

    if urban_data is not None:

        urban_mask = (
            create_binary_exclusion_mask(
                urban_data,
                "URBAN",
                nodata_value=urban_nodata,
                exclude_nodata=False
            )
        )

        urban_inside = (
                parcel_mask &
                urban_mask
        )

        print(
            "Pixeli urban în parcelă:",
            np.count_nonzero(
                urban_inside
            )
        )

        allowed_mask &= (
            ~urban_mask
        )

    else:

        print(
            "[INFO] urban_data = None."
        )

        if (
                detected_scheme ==
                "worldcover"
        ):

            print(
                "Built-up este totuși "
                "analizat prin clasa "
                "WorldCover 50."
            )

        elif (
                detected_scheme ==
                "corine"
        ):

            print(
                "Zonele urbane sunt "
                "totuși analizate prin "
                "clasele CORINE."
            )

    # ========================================================
    # 7. GARANȚIE:
    # AFARA PARCELEI = FALSE
    # ========================================================

    allowed_mask &= (
        parcel_mask
    )

    # ========================================================
    # 8. STATISTICI STRICT PE PARCELĂ
    # ========================================================

    allowed_pixels = (
        np.count_nonzero(
            allowed_mask
        )
    )

    excluded_pixels = (
            total_parcel_pixels -
            allowed_pixels
    )

    allowed_percentage = (
            allowed_pixels /
            total_parcel_pixels *
            100.0
    )

    excluded_percentage = (
            excluded_pixels /
            total_parcel_pixels *
            100.0
    )

    # ========================================================
    # 9. PĂDURE ÎN PARCELĂ
    # ========================================================

    if (
            detected_scheme ==
            "worldcover"
    ):

        forest_codes = [
            10
        ]

    else:

        forest_codes = [
            311,
            312,
            313
        ]

    forest_mask = (
            parcel_mask &
            lc_valid_mask &
            np.isin(
                lc_integer,
                forest_codes
            )
    )

    forest_pixels = (
        np.count_nonzero(
            forest_mask
        )
    )

    forest_percentage = (
            forest_pixels /
            total_parcel_pixels *
            100.0
    )

    # ========================================================
    # 10. APĂ ÎN PARCELĂ
    # ========================================================

    if (
            detected_scheme ==
            "worldcover"
    ):

        water_codes = [
            80
        ]

    else:

        water_codes = [
            511,
            512,
            521,
            522,
            523
        ]

    water_mask = (
            parcel_mask &
            lc_valid_mask &
            np.isin(
                lc_integer,
                water_codes
            )
    )

    water_pixels = (
        np.count_nonzero(
            water_mask
        )
    )

    water_percentage = (
            water_pixels /
            total_parcel_pixels *
            100.0
    )

    # ========================================================
    # 11. CONSTRUIT ÎN PARCELĂ
    # ========================================================

    if (
            detected_scheme ==
            "worldcover"
    ):

        built_codes = [
            50
        ]

    else:

        built_codes = [
            111,
            112,
            121,
            122,
            123,
            124,
            141,
            142
        ]

    built_mask = (
            parcel_mask &
            lc_valid_mask &
            np.isin(
                lc_integer,
                built_codes
            )
    )

    built_pixels = (
        np.count_nonzero(
            built_mask
        )
    )

    built_percentage = (
            built_pixels /
            total_parcel_pixels *
            100.0
    )

    # ========================================================
    # 12. REZULTAT FINAL
    # ========================================================

    print("\n")
    print("*" * 70)
    print(
        "REZULTAT FINAL - STRICT PE PARCELĂ"
    )
    print("*" * 70)

    print(
        "Land Cover detectat:",
        detected_scheme.upper()
    )

    print(
        "Clase interzise:",
        used_forbidden_classes
    )

    print(
        "Pixeli parcelă:",
        total_parcel_pixels
    )

    print(
        "Pixeli fezabili:",
        allowed_pixels
    )

    print(
        "Pixeli excluși:",
        excluded_pixels
    )

    print()

    print(
        f"TEREN FEZABIL: "
        f"{allowed_percentage:.2f}%"
    )

    print(
        f"TEREN NEFEZABIL: "
        f"{excluded_percentage:.2f}%"
    )

    print()

    print(
        f"PĂDURE ÎN PARCELĂ: "
        f"{forest_percentage:.2f}%"
    )

    print(
        f"APĂ ÎN PARCELĂ: "
        f"{water_percentage:.2f}%"
    )

    print(
        f"CONSTRUIT ÎN PARCELĂ: "
        f"{built_percentage:.2f}%"
    )

    # ========================================================
    # 13. DIAGNOSTIC RESTRICȚII
    # ========================================================

    if forest_percentage > 0:

        print(
            "\n[RESTRICȚIE] "
            "Parcela intersectează pădure "
            f"în proporție de "
            f"{forest_percentage:.2f}%."
        )

    if water_percentage > 0:

        print(
            "[RESTRICȚIE] "
            "Parcela intersectează apă "
            f"în proporție de "
            f"{water_percentage:.2f}%."
        )

    if built_percentage > 0:

        print(
            "[RESTRICȚIE] "
            "Parcela intersectează "
            "zone construite "
            f"în proporție de "
            f"{built_percentage:.2f}%."
        )

    # ========================================================
    # 14. CONTROL 100%
    # ========================================================

    if (
            allowed_percentage >=
            99.99
    ):

        print("\n")
        print("!" * 70)

        print(
            "ATENȚIE: parcela apare "
            "aproape 100% fezabilă."
        )

        print(
            "Dacă pe hartă vezi pădure, "
            "clădiri sau apă, problema "
            "este probabil la alinierea "
            "parcel_mask cu rasterul."
        )

        print("!" * 70)

    # ========================================================
    # 15. CONTROL 0%
    # ========================================================

    if (
            allowed_percentage <=
            0.01
    ):

        print(
            "\n[INFO] Parcela este "
            "practic integral nefezabilă "
            "conform restricțiilor încărcate."
        )

    print("\n")
    print("*" * 70)
    print(
        "SFÂRȘIT ANALIZĂ EXCLUSION MASK"
    )
    print("*" * 70)
    print("\n")

    return allowed_mask