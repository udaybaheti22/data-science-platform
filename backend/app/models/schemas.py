from pydantic import BaseModel
from typing import List, Dict, Any

class TrainRequest(BaseModel):
    target_column: str
    feature_columns: List[str]
    test_size: float
    random_state: int
    model_name: str
    hyperparameters: Dict[str, Any]

