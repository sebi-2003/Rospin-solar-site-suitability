import numpy as np
from scipy.ndimage import distance_transform_edt

def check_buffer_zones(restricted_feature_data, pixel_resolution=30.0, buffer_distance_meters=50.0):
    """
    Verifică zonele tampon (buffer zones) față de elemente restricționate (ex: ape, căi ferate, păduri).
    Afișează mesaje de status clare în terminal dacă zona respectă distanța de siguranță sau nu.

    Parametri:
    - restricted_feature_data: Matrice binară unde 1 = element restricționat (râu, cale ferată), 0 = restul.
    - pixel_resolution: Rezoluția pixelului în metri (ex: 30m).
    - buffer_distance_meters: Distanța minimă obligatorie de siguranță (buffer) în metri.

    Returnează:
    - safe_mask: Mască binară (True unde se respectă bufferul, False unde încalcă zona de siguranță).
    """
    print(f"\n[*] Se analizează zona tampon (Buffer) de {buffer_distance_meters} metri...")

    # Verificăm dacă există elemente restricționate pe hartă
    is_feature = (restricted_feature_data == 1)

    if not np.any(is_feature):
        print("[INFO] Nu s-au detectat elemente restricționate pe această hartă. Toate zonele sunt sigure.")
        return np.ones(restricted_feature_data.shape, dtype=bool)

    # Calculăm distanța în pixeli față de cel mai apropiat element restricționat
    distance_in_pixels = distance_transform_edt(~is_feature)

    # Convertim distanța din pixeli în metri reali folosind rezoluția
    distance_in_meters = distance_in_pixels * pixel_resolution

    # Aplicăm regula bufferului: terenul este sigur doar dacă distanța este >= bufferul impus
    safe_mask = distance_in_meters >= buffer_distance_meters

    # Calculăm statistici pentru mesajele de avertizare
    violation_count = np.sum(~safe_mask)
    total_pixels = safe_mask.size
    violation_percentage = (violation_count / total_pixels) * 100

    # Afișarea mesajelor clare de status
    print("=== RAPORT STATUS ZONĂ TAMPON ===")
    if violation_count == 0:
        print("[OK] Zona este complet sigură! Toate punctele respectă distanța minimă impusă.")
    else:
        print(f"[ATENȚIE] S-au găsit încălcări ale zonei tampon!")
        print(f"   -> Procentaj suprafață blocată de buffer: {violation_percentage:.2f}%")
        print(f"   -> Mesaj: Construcția este interzisă la o distanță mai mică de {buffer_distance_meters}m față de elementul protejat.")

    return safe_mask