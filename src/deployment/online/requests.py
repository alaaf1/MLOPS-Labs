from pydantic import BaseModel
from typing import List

class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Title: str
    FamilySize: int

class InferenceRequest(BaseModel):
    inputs: List[PassengerInput]  # accepts batch of passengers