from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json

# Initialize the FastAPI application
app = FastAPI(title="FlipFlop Adaptive Agent")

# --- INPUT/OUTPUT MODELS ---
class ScoreInput(BaseModel):
    """Defines the expected input payload from the ESP32: {"score": 10}"""
    score: int = 0

class DifficultyOutput(BaseModel):
    """Defines the structure of the JSON response sent back to the ESP32: {"difficulty_index": 1}"""
    difficulty_index: int

# --- NEURAL NETWORK MODEL DEFINITION ---

# Mock Weights & Biases (Simulates a trained single-layer network)
# W: (1 input feature: score) x (3 output classes)
# B: (3 output classes)
W = np.array([[0.05], [0.15], [0.25]])
B = np.array([-1.0, -4.0, -8.0]) 

def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    e_z = np.exp(z - np.max(z)) 
    return e_z / e_z.sum(axis=0)

def predict_category(score: int) -> tuple[int, np.ndarray]:
    """
    Runs the lightweight neural network to classify player skill based on score.
    """
    # 1. Input Processing
    X = np.array([score])
    
    # 2. Linear Layer: Z = W * X + B
    Z = (W * X).flatten() + B 
    
    # 3. Activation Layer (Softmax)
    probabilities = softmax(Z)
    
    # 4. Prediction: Get the index of the highest probability
    category_index = np.argmax(probabilities)
    
    return category_index, probabilities

# --- FASTAPI ENDPOINT ---

@app.post("/adaptive_logic", response_model=DifficultyOutput)
async def adaptive_logic(data: ScoreInput):
    """Receives score, predicts player category using NN, and returns difficulty index."""
    try:
        score = data.score
        
        # 1. Predict Player Category (0, 1, or 2)
        category_index, probabilities = predict_category(score)
        
        # 2. Map Category to Difficulty Index (Fixing NumPy type)
        # CRITICAL FIX: Convert NumPy integer type to standard Python integer (int()) 
        # to prevent the "int64 is not JSON serializable" error.
        new_difficulty_index = int(category_index) 
        
        # Log the decision
        log_data = {
            "score": score,
            "difficulty_index": new_difficulty_index,
            "probabilities": probabilities.tolist() # Convert NumPy array to list for logging
        }
        print(f"Decision Log: {json.dumps(log_data)}")
        
        # 3. Return the index
        return DifficultyOutput(difficulty_index=new_difficulty_index)

    except Exception as e:
        # Raise HTTP exception for error handling on client side
        print(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"difficulty_index": 1, "error": "Internal AI server error"}
        )

# --- ROOT (Health Check) ---
@app.get("/")
def read_root():
    return {"status": "FlipFlop Adaptive Agent is Online"}
