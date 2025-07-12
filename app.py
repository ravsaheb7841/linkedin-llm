import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Page setup
st.set_page_config(page_title="LinkedIn Post Generator", layout="centered")
st.title("🤖 Local LinkedIn Post Generator")

# Load model + tokenizer with cache
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./linkedin_model")
    model = AutoModelForCausalLM.from_pretrained("./linkedin_model")
    return tokenizer, model

tokenizer, model = load_model()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# UI inputs
theme = st.text_input("Enter a theme for your LinkedIn post:", placeholder="e.g. I'm happy to share my first internship")
max_tokens = st.slider("Select max words to generate", min_value=10, max_value=50, value=20)

# SINGLE BUTTON (Only once)
if st.button("Generate Post", key="generate_button"):
    if theme.strip() == "":
        st.warning("⚠️ Please enter a valid theme.")
    else:
        # Tokenize and generate
        inputs = tokenizer(theme, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Output
        st.markdown("### ✅ Generated LinkedIn Post")
        st.success(result)
