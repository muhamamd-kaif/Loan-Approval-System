import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIG ---
st.set_page_config(page_title="LoanGuard AI | Premium Predictor", page_icon="🏦", layout="wide")

# --- HIGH-END DARK THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp { background-color: #0b0e14; color: #e1e1e1; font-family: 'Inter', sans-serif; }
    
    /* Header Glow */
    .main-title {
        font-size: 50px; font-weight: 800; background: linear-gradient(90deg, #00d4ff, #007bff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; letter-spacing: -1px;
    }
    
    /* Custom Inputs - Improved Spacing */
    .stNumberInput input {
        background-color: #1a1f29 !important;
        color: white !important;
        border: 1px solid #333a45 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #1a1f29 !important;
        color: white !important;
        border: 1px solid #333a45 !important;
        border-radius: 10px !important;
    }

    /* Label Spacing */
    label p {
        margin-bottom: 5px !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    
    /* Button Animation */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 4em;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white; font-weight: 700; border: none;
        box-shadow: 0 10px 20px rgba(0, 123, 255, 0.2); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 15px 30px rgba(0, 123, 255, 0.4); }
    
    /* Result Card Styles */
    .result-card {
        padding: 40px; border-radius: 24px; text-align: center;
        margin: 30px 0; font-size: 32px; font-weight: 800;
        animation: slideIn 0.6s ease-out;
    }
    .approve { 
        background: linear-gradient(145deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.05));
        color: #2ecc71; border: 2px solid #2ecc71;
        box-shadow: 0 20px 40px rgba(46, 204, 113, 0.15);
    }
    .reject { 
        background: linear-gradient(145deg, rgba(220, 53, 69, 0.1), rgba(220, 53, 69, 0.05));
        color: #e74c3c; border: 2px solid #e74c3c;
        box-shadow: 0 20px 40px rgba(231, 76, 60, 0.15);
    }
    
    @keyframes slideIn { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    
    /* Metrics Box */
    .metric-container { background: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    
    /* Sidebar Cleanup */
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d; width: 350px !important; }
    .about-header { font-size: 24px; font-weight: 800; color: #58a6ff; margin-bottom: 20px; }
    .dev-tag { background: #238636; color: white; padding: 5px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; display: inline-block; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: OPTIMIZED ABOUT SECTION ---
with st.sidebar:
    st.markdown('<div class="about-header">🏦 LoanGuard AI</div>', unsafe_allow_html=True)
    
    st.markdown("### 👨‍💻 Project Developers")
    st.markdown('<div class="dev-tag">M.Kaif</div> <div class="dev-tag">Laiba Wazir</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("🎯 Our Mission", expanded=True):
        st.write("""
        Providing financial clarity through machine learning. This app predicts loan repayment reliability 
        to help institutions make faster, data-driven decisions.
        """)
        
    with st.expander("🧠 The Technology", expanded=False):
        st.write("""
        - **Model**: Extreme Gradient Boosting (XGBoost)
        - **Data**: Trained on 100k+ historical credit records.
        - **Logic**: Analyzes 20 unique financial features simultaneously.
        """)
        
    with st.expander("📊 How to Read Results", expanded=False):
        st.write("""
        - **LTI**: Loan amount vs your annual pay. Lower is safer.
        - **DTI**: Monthly bills vs monthly income. Should be < 40%.
        - **Utilization**: How much of your credit limit is used. Should be < 30%.
        """)
    
    st.markdown("---")
    st.caption("Final Project Version 1.0")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'Models', 'xgb_model.pkl')
    scaler_path = os.path.join(base_dir, 'Models', 'scaler.pkl')
    try:
        m = joblib.load(model_path)
        s = joblib.load(scaler_path)
        return m, s
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_assets()

def get_input_df(data):
    job_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    
    cs = data['credit_score']
    if cs > 850: cs = cs / 10
    cs = np.clip(cs, 300, 850)
    
    dti = data['debt'] / (data['income'] / 12) if data['income'] > 0 else 0
    util = data['balance'] / (data['max_credit'] + 1)
    lti = data['loan'] / data['income'] if data['income'] > 0 else 0
    
    features = {
        'Credit Score': cs, 'Loan_to_Income': lti, 'Current Loan Amount': data['loan'],
        'Credit_Utilization': util, 'DTI_Ratio': dti, 'Maximum Open Credit': data['max_credit'],
        'Years of Credit History': data['history'], 'Current Credit Balance': data['balance'],
        'Monthly Debt': data['debt'], 'Annual Income': data['income'],
        'Number of Open Accounts': data['accounts'], 'Years in current job': job_map.get(data['job'], 0),
        'Term': 1 if data['term'] == 'Short Term' else 0, 'Number of Credit Problems': data['problems'],
        'Purpose_debt consolidation': 1 if data['purpose'] == 'Debt Consolidation' else 0,
        'Home Ownership_Rent': 1 if data['home'] == 'Rent' else 0,
        'Home Ownership_Home Mortgage': 1 if data['home'] == 'Home Mortgage' else 0,
        'Bankruptcies': data['bankruptcies'], 'Purpose_other': 1 if data['purpose'] not in ['Debt Consolidation'] else 0,
        'Home Ownership_Own Home': 1 if data['home'] == 'Own Home' else 0
    }
    
    cols = ['Credit Score', 'Loan_to_Income', 'Current Loan Amount', 'Credit_Utilization', 'DTI_Ratio', 
            'Maximum Open Credit', 'Years of Credit History', 'Current Credit Balance', 'Monthly Debt', 
            'Annual Income', 'Number of Open Accounts', 'Years in current job', 'Term', 
            'Number of Credit Problems', 'Purpose_debt consolidation', 'Home Ownership_Rent', 
            'Home Ownership_Home Mortgage', 'Bankruptcies', 'Purpose_other', 'Home Ownership_Own Home']
    
    return pd.DataFrame([features])[cols]

# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">LoanGuard AI</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e; font-size:18px; margin-top:-10px;'>Next-Gen Credit Risk Intelligence</p>", unsafe_allow_html=True)
st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Applicant Profile")
    income = st.number_input("Annual Income ($)", value=75000, step=5000)
    job = st.selectbox("Years in Job", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], index=10)
    home = st.selectbox("Housing", ['Home Mortgage', 'Rent', 'Own Home'])

with col2:
    st.markdown("#### 💵 Loan Request")
    loan = st.number_input("Loan Amount ($)", value=20000, step=1000)
    term = st.selectbox("Term", ['Short Term', 'Long Term'])
    purpose = st.selectbox("Purpose", ['Debt Consolidation', 'Business Loan', 'Home Improvement', 'Other'])
    debt = st.number_input("Monthly Debts ($)", value=1200, step=100)

with col3:
    st.markdown("#### 📊 Credit Health")
    credit_score = st.number_input("Credit Score", value=720, min_value=300, max_value=850)
    history = st.number_input("History (Years)", value=15.0)
    accounts = st.number_input("Open Accounts", value=10)
    max_credit = st.number_input("Credit Limit ($)", value=40000)

with st.expander("🛠️ Advanced Credit Details"):
    ca, cb, cc = st.columns(3)
    with ca: balance = st.number_input("Current Balance ($)", value=8000)
    with cb: problems = st.number_input("Public Credit Issues", value=0)
    with cc: bankruptcies = st.number_input("Bankruptcies", value=0)

st.write("")
if st.button("EXECUTE RISK ASSESSMENT"):
    if model and scaler:
        input_data = {
            'income': income, 'job': job, 'home': home, 'loan': loan, 'term': term, 
            'purpose': purpose, 'debt': debt, 'credit_score': credit_score, 
            'history': history, 'accounts': accounts, 'max_credit': max_credit, 
            'balance': balance, 'problems': problems, 'bankruptcies': bankruptcies
        }
        df_final = get_input_df(input_data)
        scaled_data = scaler.transform(df_final)
        # Apply the custom 0.71 strict threshold from the notebook
        probability = model.predict_proba(scaled_data)[0][1]
        
        if probability >= 0.71:
            st.markdown('<div class="result-card approve">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
            st.caption(f"Confidence Score: {probability:.1%}")
        else:
            st.markdown('<div class="result-card reject">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
            st.caption(f"Confidence Score: {probability:.1%} (Requires ≥ 71%)")
        
        st.write("")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Loan-to-Income", f"{df_final['Loan_to_Income'][0]:.2f}")
        with mc2:
            st.metric("Debt-to-Income", f"{df_final['DTI_Ratio'][0]:.1%}")
        with mc3:
            st.metric("Credit Utilization", f"{df_final['Credit_Utilization'][0]:.1%}")
    else:
        st.error("Model Engine Offline")
