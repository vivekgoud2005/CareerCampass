"""
CareerCompass - Agentic Career Counseling Companion
Built with Streamlit + IBM watsonx.ai (Granite model)

HOW TO GET YOUR IBM CREDENTIALS (see README.md for full steps):
1. Create IBM Cloud Lite account: https://cloud.ibm.com
2. Create a watsonx.ai project, provision Watson Machine Learning (Lite)
3. Go to Prompt Lab -> View Code -> copy your API_KEY, PROJECT_ID, and URL
4. Paste them into the sidebar when you run this app (or set as environment variables)
"""

import streamlit as st
import requests
import json

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="CareerCompass - AI Career Counselor", page_icon="🧭", layout="wide")

# ---------------------------------------------------------
# HELPER: Get IBM IAM Access Token from API Key
# ---------------------------------------------------------
def get_iam_token(api_key: str) -> str:
    """Exchanges your IBM Cloud API key for a temporary access token."""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


# ---------------------------------------------------------
# HELPER: Call Granite model on watsonx.ai
# ---------------------------------------------------------
def call_granite(prompt: str, api_key: str, project_id: str, url: str, model_id: str) -> str:
    """Sends a prompt to the Granite model and returns the generated text."""
    token = get_iam_token(api_key)

    endpoint = f"{url}/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "min_new_tokens": 50,
            "repetition_penalty": 1.1,
        },
        "model_id": model_id,
        "project_id": project_id,
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(body))
    response.raise_for_status()
    result = response.json()
    return result["results"][0]["generated_text"]


# ---------------------------------------------------------
# HELPER: Build the prompt from student data
# ---------------------------------------------------------
def build_prompt(name, strengths, weak_areas, interests, market_note):
    return f"""You are an expert career counselor AI. Analyze the student profile below and suggest 3 suitable career pathways.
For each pathway, give: (1) the career name, (2) why it fits based on their academic strengths and interests,
(3) one key skill they should develop next, and (4) a relevant current industry trend.

Student Name: {name}
Academic Strengths: {strengths}
Weaker Areas: {weak_areas}
Interests / Hobbies: {interests}
Known Market Trend Context: {market_note}

Respond in a clear, encouraging tone, formatted as a numbered list of 3 career paths.
"""


# ---------------------------------------------------------
# HELPER: Offline demo fallback (used if no API key provided)
# ---------------------------------------------------------
def demo_response(name, strengths, interests):
    return f"""**(Demo Mode - no IBM API key entered yet)**

Here's a sample of what Granite would generate for **{name}**:

1. **Data Analyst / Data Scientist**
   - *Why it fits:* Your strength in {strengths} combined with an interest in {interests} maps well to data-driven roles.
   - *Skill to build next:* SQL and basic Python/pandas.
   - *Trend:* Demand for entry-level data roles has grown steadily as companies digitize operations.

2. **Product Management (Tech)**
   - *Why it fits:* Strong analytical thinking plus curiosity about {interests} suggests you enjoy connecting ideas to real outcomes.
   - *Skill to build next:* Basic wireframing and user research fundamentals.
   - *Trend:* Companies increasingly hire "APM" (Associate Product Manager) tracks for fresh graduates.

3. **UX Research / Design**
   - *Why it fits:* If {interests} involves understanding people or systems, this blends creativity with structured thinking.
   - *Skill to build next:* Learn Figma and basic usability testing methods.
   - *Trend:* Human-centered design roles are expanding across fintech and healthtech.

*(Enter your real IBM watsonx.ai credentials in the sidebar to replace this with live Granite-generated output.)*
"""


# ---------------------------------------------------------
# SIDEBAR: IBM Credentials
# ---------------------------------------------------------
st.sidebar.title("⚙️ IBM watsonx.ai Setup")
st.sidebar.markdown("Paste your credentials from **Prompt Lab → View Code**.")

api_key = st.sidebar.text_input("IBM Cloud API Key", type="password")
project_id = st.sidebar.text_input("watsonx.ai Project ID")
region_url = st.sidebar.selectbox(
    "Region URL",
    ["https://us-south.ml.cloud.ibm.com", "https://eu-de.ml.cloud.ibm.com", "https://au-syd.ml.cloud.ibm.com"],
)
model_id = st.sidebar.selectbox(
    "Granite Model",
    ["ibm/granite-13b-instruct-v2", "ibm/granite-3-8b-instruct"],
)

st.sidebar.markdown("---")
st.sidebar.info("No credentials yet? The app will run in **Demo Mode** using sample output so you can still test the UI and logic.")

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("🧭 CareerCompass")
st.caption("An agentic career counseling companion — powered by IBM Granite (watsonx.ai)")

st.markdown("### Step 1: Tell us about the student")

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Student Name", "Aarav")
    strengths = st.text_area("Academic Strengths (subjects/skills)", "Mathematics, Physics, Logical reasoning")
    weak_areas = st.text_area("Weaker Areas", "Public speaking, Creative writing")
with col2:
    interests = st.text_area("Interests / Hobbies", "Solving puzzles, building small apps, gaming")
    market_note = st.text_area(
        "Known Market Trend Context (optional)",
        "AI/ML and data roles are growing fast in the region",
    )

st.markdown("### Step 2: Generate career guidance")

if st.button("🔍 Get Career Recommendations", type="primary"):
    with st.spinner("Analyzing profile and consulting Granite..."):
        prompt = build_prompt(name, strengths, weak_areas, interests, market_note)

        if api_key and project_id:
            try:
                output = call_granite(prompt, api_key, project_id, region_url, model_id)
                st.success("Recommendations generated using live IBM Granite model ✅")
                st.markdown(output)
            except Exception as e:
                st.error(f"Could not reach watsonx.ai: {e}")
                st.info("Showing demo output instead so you can keep testing the app:")
                st.markdown(demo_response(name, strengths, interests))
        else:
            st.warning("No API credentials entered — showing Demo Mode output.")
            st.markdown(demo_response(name, strengths, interests))

st.markdown("---")
st.markdown("### 🔁 Agentic Monitoring (Simulated)")
st.caption(
    "In the full version, this section would auto re-run analysis whenever new grades, "
    "interest survey results, or labor market data are updated — without the student "
    "needing to ask again. For this prototype, click below to simulate that trigger."
)
if st.button("Simulate: New grades uploaded → re-run analysis"):
    st.info("Agent detected updated academic data. Re-generating recommendations...")
    st.rerun()
