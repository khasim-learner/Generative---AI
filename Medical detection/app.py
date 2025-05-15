import streamlit as st
from pathlib import Path
import google.generativeai as genai
from api_key import api_key

genai.configure(api_key=api_key)

generation_config = {"temperature":0.4,"top_p":1,"top_k":32,"max_output_tokens":4096}

# apply safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with:

Your Responsibilities:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2.Findings Report: Document all observed anomalies or signs of disease. Clearly articulate them.
3.Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including...
4.Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1.Scope of Response: Only respond if the image pertains to human health issues.
2.Clarity of Image: In cases where the image quality impedes clear analysis, note that certain...
3.Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before maki...
4.Your insights are invaluable in guiding clinical decisions. Please proceed with the analys...

pls provide me an output response with these 4 heading detailed Analysis, Finding reports, Recomendations and Next Steps, Treatment Suggestions.

"""
model = genai.GenerativeModel(model_name="gemini-1.5-flash",generation_config=generation_config,safety_settings=safety_settings)

st.set_page_config(page_title="Vital Image Analytics",page_icon=":robot:")

st.title("Vital ❤️ Image 📷 Analytics 📊👨‍⚕️")

st.subheader("An Application to help get details from Medical Pictures")

upload_file = st.file_uploader("Upload the Medical Images",type = ["png","jpg","jpeg"])

submit_button = st.button("Generate the Analysis")

if submit_button:
    image_data = upload_file.getvalue()
    
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
     
    prompt_parts = [image_parts[0], system_prompt]

    response = model.generate_content(prompt_parts)

    st.write(response.text)  # show the result in Streamlit app
