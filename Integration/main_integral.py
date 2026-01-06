from integration import Integrator
import torch
import math
import time

def integrate():
    print("Demarrage de l'integration")
    
    def integrate_function(x):
        return torch.sin(x) #exemple
    
    integrator = Integrator(integrate_function)
    
    a = 0
    b = math.pi
    n = 10000
    
    print(f"\nCalcul de l'integrale de sin(x) entre {a} et {b:.4f}")
    print(f"nombre de subdivisions : {n}\n\n")
    
    
    start_time = time.time()
    res_trap = integrator.trapezoidal(a, b, n)
    end_time = time.time()
    
    print(f"Méthode Trapèzes : {res_trap:.6f}")
    print(f"Temps d'exécution: {(end_time - start_time)*1000:.4f} ms")

    start_time = time.time()
    res_simpson = integrator.simpson(a, b, n)
    end_time = time.time()
    
    print(f"Méthode Simpson  : {res_simpson:.6f}")
    print(f"Temps d'exécution: {(end_time - start_time)*1000:.4f} ms")
    

    vraie_valeur = 2.0
    erreur_trap = abs(res_trap - vraie_valeur)
    erreur_simp = abs(res_simpson - vraie_valeur)
    
    print("-" * 30)
    print(f"Erreur Trapèzes : {erreur_trap:.10f}")
    print(f"Erreur Simpson  : {erreur_simp:.10f}")
    
    if erreur_simp < erreur_trap:
        print("\n=> Conclusion : La méthode de Simpson est plus précise (comme prévu en cours !).")

if __name__ == "__main__":
    integrate()