import numpy as np

def compute_mcda_score(slope_data, irradiance_data, distance_grid_data, distance_road_data):
    """
    Calculează un Scor de Suitabilitate Ponderat (MCDA) de la 0 la 100 pentru fiecare pixel.

    Ponderi recomandate (însumate fac 1.0 / 100%):
    - Rețeaua electrică: 40% (cel mai important cost economic)
    - Iradierea solară: 30% (eficiența energetică)
    - Panta terenului: 20% (ușurința lucrărilor de construcție)
    - Distanța la drum: 10% (accesul logistic)
    """
    print("\n[*] Se rulează Analiza Ponderată Multi-Criterială (MCDA - Heatmap 0-100)...")

    # 1. Normalizăm fiecare criteriu la o scală de la 0.0 la 1.0 (unde 1.0 este cel mai bun)

    # Panta: terenul de 0 grade ia scor maxim (1.0), cel de 30+ grade ia scor 0.0
    slope_score = np.clip(1.0 - (slope_data / 30.0), 0.0, 1.0)

    # Iradierea: normalizăm între un minim de 1000 și un maxim de 1400 kWh/m²/an
    irradiance_score = np.clip((irradiance_data - 1000.0) / (1400.0 - 1000.0), 0.0, 1.0)

    # Distanța la rețea: ideal la 0 metri, inacceptabil la peste 5000 metri
    grid_score = np.clip(1.0 - (distance_grid_data / 5000.0), 0.0, 1.0)

    # Distanța la drum: ideal la 0 metri, inacceptabil la peste 2000 metri
    road_score = np.clip(1.0 - (distance_road_data / 2000.0), 0.0, 1.0)

    # 2. Aplicăm ponderile (Weights) stabilite pentru inginerie
    w_grid = 0.40      # 40% Rețeaua
    w_irr = 0.30       # 30% Iradierea
    w_slope = 0.20     # 20% Panta
    w_road = 0.10      # 10% Drumul

    # 3. Calculăm scorul final ponderat și îl aducem la scara 0 - 100
    mcda_matrix = (
                          (grid_score * w_grid) +
                          (irradiance_score * w_irr) +
                          (slope_score * w_slope) +
                          (road_score * w_road)
                  ) * 100.0

    print("-> Scoruri MCDA calculate cu succes! Harta globală oferă o ierarhie clară a amplasamentelor.")
    return mcda_matrix