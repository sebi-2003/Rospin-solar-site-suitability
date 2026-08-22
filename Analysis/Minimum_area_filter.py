import numpy as np

def check_one_hectare_limit(suitability_mask, transform):
    """
    Calculează suprafața totală fezabilă. Afișează întotdeauna suprafața,
    iar dacă are sub 1 hectar, afișează suplimentar un mesaj de atenționare.
    """
    # Extragem dimensiunile pixelului din transform (în metri)
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])

    # Suprafața unui pixel în hectare (1 ha = 10.000 m²)
    pixel_area_ha = (pixel_width * pixel_height) / 10000.0

    # Numărăm totalul pixelilor fezabili de pe hartă
    total_pixels = np.sum(suitability_mask)
    total_area_ha = total_pixels * pixel_area_ha

    print("\n=== VERIFICARE SUPRAFAȚĂ TEREN ===")

    # 1. În orice caz, afișăm suprafața totală
    print(f"Suprafața totală identificată: {total_area_ha:.2f} hectare.")

    # 2. Dacă are sub un hectar, afișăm și mesajul de atenționare
    if total_area_ha < 1.0:
        print("[ATENȚIE] Terenul are sub un hectar!")

    return total_area_ha < 1.0, total_area_ha