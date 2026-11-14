from tensorflow.keras.models import load_model

# Load model
try:
    model = load_model(r'C:\Programming\FinalMed\model\VP.keras')  # Use your actual path
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)