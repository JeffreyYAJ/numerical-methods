
import torch
from typing import Union, List


class NewtonInterpolator:
    def __init__(self, x_data: Union[List[float], torch.Tensor], y_data: Union[List[float], torch.Tensor]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Conversion
        if not torch.is_tensor(x_data): x_data = torch.tensor(x_data, dtype=torch.float32)
        if not torch.is_tensor(y_data): y_data = torch.tensor(y_data, dtype=torch.float32)
        
        self.x_data = x_data.to(self.device)
        self.y_data = y_data.to(self.device)
        
        self.coefficients = self._compute_divided_differences()

    def _compute_divided_differences(self) -> torch.Tensor:
        
        n = len(self.x_data)
        coefs = self.y_data.clone()
        
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                numerateur = coefs[i] - coefs[i-1]
                denominateur = self.x_data[i] - self.x_data[i-j]
                
                coefs[i] = numerateur / denominateur
                
        return coefs

    def predict(self, x_query: Union[List[float], torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(x_query):
            x_query = torch.tensor(x_query, dtype=torch.float32)
        x_query = x_query.to(self.device)
        
        n = len(self.x_data) - 1
        p = self.coefficients[n]
        
        for k in range(1, n + 1):
            p = self.coefficients[n - k] + (x_query - self.x_data[n - k]) * p
            
        return p