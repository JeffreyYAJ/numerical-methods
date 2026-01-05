import torch
import matplotlib.pyplot as plt
from interpolation import LagrangeInterpolator 

def run_demo():
    print("--- Démarrage de la démo d'Interpolation ---")
    
    x_points = [0.0, 1.0, 2.0, 3.5]
    y_points = [0.0, 2.0, 1.0, 4.0]

    try:
        model = LagrangeInterpolator(x_points, y_points)
    except ValueError as e:
        print(e)
        return

    x_test = torch.linspace(-0.5, 4.0, 100)
    
    
    y_test = model.predict(x_test)

    print("Génération du graphique...")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x_points, y_points, color='red', label='Données connues (Mesures)', zorder=5)
    
    plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Polynôme de Lagrange (PyTorch)', color='blue')
    
    plt.title("Interpolation Numérique avec Accélération GPU")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    run_demo()