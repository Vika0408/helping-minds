import os
import google.generativeai as genai

# Load the Gemini API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Environment variable 'GEMINI_API_KEY' not set.")
    exit()

# Configure the Gemini client
genai.configure(api_key=api_key)

try:
    # Use a supported model from your list
    model = genai.GenerativeModel(model_name='models/gemini-2.5-flash-preview-05-20')
    
    # Make a simple content generation request
    response = model.generate_content("Hello Gemini! Can you hear me?")
    
    print("✅ Gemini API Response:")
    print(response.text)

except Exception as e:
    print("❌ Error communicating with Gemini API:")
    print(e)
