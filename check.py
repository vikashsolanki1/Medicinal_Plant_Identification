import google.generativeai as genai

# Configure Gemini API with your API key
genai.configure(api_key="AIzaSyCHWQjXQhPk2SRtypTfTiR3-arysC2aowY")

# Try to list the available models
try:
    available_models = list(genai.list_models())  # Convert the generator to a list
    print("Available Models:", available_models)
except Exception as e:
    print(f"Error fetching models: {str(e)}")
