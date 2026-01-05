import torch
from typing import Union, List

class LagrangeInterpolator:
    def __init__(self, x_data: Union[List[float], torch.Tensor], y_data: Union[List[float], torch.Tensor]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Initialisation sur {self.device}")

        if not torch.is_tensor(x_data):
            x_data = torch.tensor(x_data, dtype=torch.float32)
        if not torch.is_tensor(y_data):
            y_data = torch.tensor(y_data, dtype=torch.float32)

        self.x_data = x_data.to(self.device)
        self.y_data = y_data.to(self.device)
        
        if self.x_data.shape != self.y_data.shape:
            raise ValueError(f"Erreur: x_data et y_data doivent avoir la même taille. Reçu {self.x_data.shape} et {self.y_data.shape}")

    def _compute_basis_polynomial(self, i: int, x_query: torch.Tensor) -> torch.Tensor:
        
        L_i = torch.ones_like(x_query).to(self.device)
        n = len(self.x_data)

        for j in range(n):
            if i != j:
                numerator = x_query - self.x_data[j]
                denominator = self.x_data[i] - self.x_data[j]
                if denominator == 0:
                    raise ZeroDivisionError("Deux points x_data sont identiques, impossible d'interpoler.")
                
                L_i = L_i * (numerator / denominator)
        
        return L_i

    def predict(self, x_query: Union[List[float], torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(x_query):
            x_query = torch.tensor(x_query, dtype=torch.float32)
        
        x_query = x_query.to(self.device)
        
        y_interpolated = torch.zeros_like(x_query).to(self.device)
        n = len(self.x_data)
        
        for i in range(n):
            basis = self._compute_basis_polynomial(i, x_query)
            term = self.y_data[i] * basis
            y_interpolated += term
            
        return y_interpolated