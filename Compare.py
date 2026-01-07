import torch
import time
import matplotlib.pyplot as plt
from interpolation_lagrange import LagrangeInterpolator
from Interpolation_newton import NewtonInterpolator

#
#       AI Generated
#

def run_comparison():
    print("--- MATCH : LAGRANGE vs NEWTON ---")
    
    # 1. Génération de données (un peu plus costaud : 20 points)
    # Fonction : y = x * sin(x)
    x_train = torch.linspace(0, 10, 20)
    y_train = x_train * torch.sin(x_train)
    
    # Points de test (beaucoup de points pour lisser la courbe)
    x_test = torch.linspace(0, 10, 1000)
    
    print(f"Entraînement sur {len(x_train)} points. Prédiction sur {len(x_test)} points.")
    
    # --- ROUND 1 : LAGRANGE ---
    print("\n1. Exécution Lagrange...")
    start_l = time.time()
    
    model_l = LagrangeInterpolator(x_train, y_train)
    y_pred_l = model_l.predict(x_test)
    
    time_l = time.time() - start_l
    print(f"-> Temps Lagrange : {time_l:.4f} secondes")

    # --- ROUND 2 : NEWTON ---
    print("\n2. Exécution Newton...")
    start_n = time.time()
    
    model_n = NewtonInterpolator(x_train, y_train)
    y_pred_n = model_n.predict(x_test)
    
    time_n = time.time() - start_n
    print(f"-> Temps Newton   : {time_n:.4f} secondes")
    
    # --- ANALYSE DES RÉSULTATS ---
    
    # Vérification mathématique : La différence doit être proche de 0
    difference = torch.abs(y_pred_l - y_pred_n).mean().item()
    print(f"\nDifférence moyenne entre les deux courbes : {difference:.10f}")
    
    if difference < 1e-4:
        print("✅ SUCCÈS : Les deux méthodes convergent vers le même polynôme.")
    else:
        print("⚠️ ATTENTION : Divergence numérique détectée.")

    # --- VISUALISATION ---
    plt.figure(figsize=(12, 6))
    
    # On trace les points d'origine
    plt.scatter(x_train.cpu(), y_train.cpu(), c='black', label='Points connus', zorder=5)
    
    # On trace Lagrange (Ligne continue)
    plt.plot(x_test.cpu(), y_pred_l.cpu(), 'b-', linewidth=3, alpha=0.5, label='Lagrange')
    
    # On trace Newton (Pointillés par dessus pour montrer que c'est pareil)
    plt.plot(x_test.cpu(), y_pred_n.cpu(), 'r--', label='Newton')
    
    plt.title("Comparaison Interpolation : Lagrange vs Newton")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_comparison()