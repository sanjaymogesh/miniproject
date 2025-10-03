from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# --- NEURAL NETWORK MODEL DEFINITION ---

# Mock Weights & Biases (Trained on simplified data: input=score)
# Weights and biases are pre-set, simulating a trained model.
# W: (1 input feature: score) x (3 output classes: 0, 1, 2)
# B: (3 output classes)
W = np.array([[0.05], [0.15], [0.25]])
B = np.array([-1.0, -4.0, -8.0])

def softmax(z):
    """Softmax activation function for classification output."""
    e_z = np.exp(z - np.max(z)) # Subtract max for numerical stability
    return e_z / e_z.sum(axis=0)

def predict_category(score):
    """
    Runs the lightweight neural network to classify player skill.
    Input: Score (1 feature). Output: Probability for 3 categories.
    """
    # 1. Input Processing (Input must be 1x1 array)
    X = np.array([score])
    
    # 2. Linear Layer: Z = W * X + B
    # W is (1x3), X is (1x1). We must transpose/reshape for matrix multiplication.
    # W.T is (3x1). X is (1x1). Result is (3x1).
    # We use element-wise multiplication here for simplicity, treating score as a scalar.
    Z = (W * X).flatten() + B 
    
    # 3. Activation Layer (Softmax)
    probabilities = softmax(Z)
    
    # 4. Prediction: Get the index of the highest probability
    category_index = np.argmax(probabilities)
    
    return category_index, probabilities

# --- MAPPING: CATEGORY INDEX to DIFFICULTY INDEX ---
# The NN output category (0, 1, 2) directly maps to the game difficulty index (0, 1, 2)
# 0 (Struggling) -> 0 (EASY)
# 1 (Steady)     -> 1 (MEDIUM)
# 2 (Master)     -> 2 (HARD)

@app.route('/adaptive_logic', methods=['POST'])
def adaptive_logic():
    """Endpoint that receives the current score and returns the appropriate difficulty index."""
    try:
        data = request.get_json()
        score = data.get('score', 0)
        
        # 1. Predict Player Category (0, 1, or 2)
        category_index, probabilities = predict_category(score)
        
        # 2. Map Category to Difficulty Index
        new_difficulty_index = category_index 
        
        print(f"Score: {score} | NN Output: {probabilities} | Predicted Category Index: {category_index}")
        
        # 3. Return the index in a simple JSON format
        return jsonify({"difficulty_index": new_difficulty_index}), 200

    except Exception as e:
        print(f"Error processing request: {e}")
        # Return default Medium difficulty (index 1) on error
        return jsonify({"difficulty_index": 1, "error": str(e)}), 500

if __name__ == '__main__':
    # Use gunicorn on Render, but Flask for local testing
    app.run(host='0.0.0.0', port=5000)
