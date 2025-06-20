import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Treatment Planner",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .treatment-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diabetes dataset with synthetic treatment data"""
    
    # Create a comprehensive diabetes dataset with treatment outcomes
    np.random.seed(42)
    n_patients = 2000
    
    # Generate patient features
    data = {
        'age': np.random.normal(55, 15, n_patients).astype(int),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'bmi': np.random.normal(28, 6, n_patients),
        'glucose_level': np.random.normal(140, 40, n_patients),
        'hba1c': np.random.normal(8.5, 2, n_patients),
        'blood_pressure_systolic': np.random.normal(135, 20, n_patients),
        'blood_pressure_diastolic': np.random.normal(85, 15, n_patients),
        'cholesterol': np.random.normal(200, 50, n_patients),
        'diabetes_duration': np.random.exponential(5, n_patients),
        'family_history': np.random.choice([0, 1], n_patients, p=[0.4, 0.6]),
        'smoking': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'exercise_hours_week': np.random.exponential(3, n_patients),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['age'] = np.clip(df['age'], 18, 90)
    df['bmi'] = np.clip(df['bmi'], 15, 50)
    df['glucose_level'] = np.clip(df['glucose_level'], 70, 300)
    df['hba1c'] = np.clip(df['hba1c'], 5, 15)
    df['blood_pressure_systolic'] = np.clip(df['blood_pressure_systolic'], 90, 200)
    df['blood_pressure_diastolic'] = np.clip(df['blood_pressure_diastolic'], 60, 120)
    df['cholesterol'] = np.clip(df['cholesterol'], 100, 400)
    df['diabetes_duration'] = np.clip(df['diabetes_duration'], 0, 30)
    df['exercise_hours_week'] = np.clip(df['exercise_hours_week'], 0, 20)
    
    # Define treatment options
    treatments = ['Metformin', 'Insulin', 'Sulfonylurea', 'DPP4_Inhibitor', 'Lifestyle_Only']
    
    # Assign treatments based on patient characteristics (realistic logic)
    def assign_treatment(row):
        if row['hba1c'] > 10 or row['glucose_level'] > 200:
            return np.random.choice(['Insulin', 'Metformin'], p=[0.7, 0.3])
        elif row['hba1c'] > 8:
            return np.random.choice(['Metformin', 'Sulfonylurea', 'DPP4_Inhibitor'], p=[0.5, 0.3, 0.2])
        elif row['bmi'] > 30:
            return np.random.choice(['Metformin', 'DPP4_Inhibitor'], p=[0.7, 0.3])
        else:
            return np.random.choice(['Lifestyle_Only', 'Metformin'], p=[0.4, 0.6])
    
    df['treatment'] = df.apply(assign_treatment, axis=1)
    
    # Generate treatment outcomes based on patient characteristics and treatment
    def calculate_outcome(row):
        base_success = 0.5
        
        # Treatment effectiveness
        treatment_effectiveness = {
            'Metformin': 0.7, 'Insulin': 0.8, 'Sulfonylurea': 0.65,
            'DPP4_Inhibitor': 0.6, 'Lifestyle_Only': 0.4
        }
        
        success_prob = treatment_effectiveness[row['treatment']]
        
        # Adjust based on patient factors
        if row['age'] < 40:
            success_prob += 0.1
        elif row['age'] > 70:
            success_prob -= 0.1
            
        if row['bmi'] > 35:
            success_prob -= 0.15
        elif row['bmi'] < 25:
            success_prob += 0.1
            
        if row['exercise_hours_week'] > 5:
            success_prob += 0.15
            
        if row['smoking']:
            success_prob -= 0.1
            
        if row['family_history']:
            success_prob -= 0.05
            
        success_prob = np.clip(success_prob, 0.1, 0.95)
        return 1 if np.random.random() < success_prob else 0
    
    df['treatment_success'] = df.apply(calculate_outcome, axis=1)
    
    # Add some complications/side effects
    def get_side_effects(row):
        if row['treatment'] == 'Insulin':
            return 'Weight gain, Hypoglycemia risk'
        elif row['treatment'] == 'Metformin':
            return 'GI upset, Vitamin B12 deficiency'
        elif row['treatment'] == 'Sulfonylurea':
            return 'Hypoglycemia, Weight gain'
        elif row['treatment'] == 'DPP4_Inhibitor':
            return 'Joint pain, Pancreatitis (rare)'
        else:
            return 'None'
    
    df['side_effects'] = df.apply(get_side_effects, axis=1)
    
    return df

@st.cache_resource
def train_similarity_model(df):
    """Train the patient similarity model"""
    
    # Prepare features for similarity matching
    feature_cols = ['age', 'bmi', 'glucose_level', 'hba1c', 'blood_pressure_systolic', 
                   'blood_pressure_diastolic', 'cholesterol', 'diabetes_duration', 
                   'family_history', 'smoking', 'exercise_hours_week']
    
    # Add gender encoding
    df_model = df.copy()
    df_model['gender_encoded'] = LabelEncoder().fit_transform(df_model['gender'])
    feature_cols.append('gender_encoded')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[feature_cols])
    
    # Train KNN model
    knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
    knn_model.fit(X_scaled)
    
    return knn_model, scaler, feature_cols

def get_treatment_recommendations(patient_data, df, knn_model, scaler, feature_cols):
    """Get treatment recommendations for a patient"""
    
    # Prepare patient data
    patient_features = []
    for col in feature_cols:
        if col == 'gender_encoded':
            patient_features.append(1 if patient_data['gender'] == 'M' else 0)
        else:
            patient_features.append(patient_data[col])
    
    # Scale patient features
    patient_scaled = scaler.transform([patient_features])
    
    # Find similar patients
    distances, indices = knn_model.kneighbors(patient_scaled)
    similar_patients = df.iloc[indices[0]]
    
    # Calculate treatment effectiveness from similar patients
    treatment_stats = {}
    treatments = similar_patients['treatment'].unique()
    
    for treatment in treatments:
        treatment_patients = similar_patients[similar_patients['treatment'] == treatment]
        if len(treatment_patients) > 0:
            success_rate = treatment_patients['treatment_success'].mean()
            count = len(treatment_patients)
            side_effects = treatment_patients['side_effects'].iloc[0]
            treatment_stats[treatment] = {
                'success_rate': success_rate,
                'count': count,
                'side_effects': side_effects
            }
    
    # Apply safety rules
    contraindications = []
    if patient_data['age'] > 75 and 'Sulfonylurea' in treatment_stats:
        contraindications.append('Sulfonylurea not recommended for age > 75 (hypoglycemia risk)')
        del treatment_stats['Sulfonylurea']
    
    if patient_data['glucose_level'] > 250 and 'Lifestyle_Only' in treatment_stats:
        contraindications.append('Lifestyle only insufficient for severe hyperglycemia')
        del treatment_stats['Lifestyle_Only']
    
    # Sort treatments by success rate
    sorted_treatments = sorted(treatment_stats.items(), 
                             key=lambda x: x[1]['success_rate'], reverse=True)
    
    return sorted_treatments, similar_patients, contraindications

def main():
    st.markdown('<div class="main-header">🏥 AI-Powered Treatment Planning System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This system uses machine learning to recommend personalized diabetes treatments based on 
    patient similarity matching and historical treatment outcomes.
    """)
    
    # Load data
    with st.spinner("Loading treatment database..."):
        df = load_and_prepare_data()
        knn_model, scaler, feature_cols = train_similarity_model(df)
    
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("Please enter patient details:")
    
    # Patient input form
    with st.sidebar:
        st.markdown("### Basic Demographics")
        age = st.slider("Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["M", "F"])
        
        st.markdown("### Physical Measurements")
        bmi = st.number_input("BMI", 15.0, 50.0, 28.0, step=0.1)
        
        st.markdown("### Lab Results")
        glucose_level = st.number_input("Glucose Level (mg/dL)", 70, 300, 140)
        hba1c = st.number_input("HbA1c (%)", 5.0, 15.0, 8.5, step=0.1)
        
        col1, col2 = st.columns(2)
        with col1:
            bp_sys = st.number_input("BP Systolic", 90, 200, 135)
        with col2:
            bp_dia = st.number_input("BP Diastolic", 60, 120, 85)
        
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
        
        st.markdown("### Medical History")
        diabetes_duration = st.number_input("Diabetes Duration (years)", 0.0, 30.0, 2.0, step=0.5)
        family_history = st.checkbox("Family History of Diabetes")
        smoking = st.checkbox("Current Smoker")
        exercise_hours = st.number_input("Exercise Hours/Week", 0.0, 20.0, 2.0, step=0.5)
    
    # Create patient data dictionary
    patient_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'glucose_level': glucose_level,
        'hba1c': hba1c,
        'blood_pressure_systolic': bp_sys,
        'blood_pressure_diastolic': bp_dia,
        'cholesterol': cholesterol,
        'diabetes_duration': diabetes_duration,
        'family_history': int(family_history),
        'smoking': int(smoking),
        'exercise_hours_week': exercise_hours
    }
    
    if st.sidebar.button("🔍 Get Treatment Recommendations", type="primary"):
        
        # Get recommendations
        recommendations, similar_patients, contraindications = get_treatment_recommendations(
            patient_data, df, knn_model, scaler, feature_cols
        )
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="section-header">🎯 Recommended Treatments</div>', 
                       unsafe_allow_html=True)
            
            if contraindications:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ **Important Contraindications:**")
                for contra in contraindications:
                    st.write(f"• {contra}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            for i, (treatment, stats) in enumerate(recommendations[:3]):
                success_rate = stats['success_rate']
                count = stats['count']
                side_effects = stats['side_effects']
                
                # Color coding based on success rate
                if success_rate >= 0.7:
                    color = "#28a745"  # Green
                elif success_rate >= 0.5:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div class="treatment-card" style="border-left-color: {color};">
                    <h3>#{i+1} {treatment.replace('_', ' ')}</h3>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p><strong>Success Rate:</strong> {success_rate:.1%}</p>
                            <p><strong>Based on:</strong> {count} similar patients</p>
                            <p><strong>Common Side Effects:</strong> {side_effects}</p>
                        </div>
                        <div style="font-size: 3rem; color: {color};">
                            {success_rate:.0%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-header">📊 Patient Profile</div>', 
                       unsafe_allow_html=True)
            
            # Risk assessment
            risk_factors = []
            if hba1c > 9:
                risk_factors.append("High HbA1c")
            if bmi > 30:
                risk_factors.append("Obesity")
            if smoking:
                risk_factors.append("Smoking")
            if bp_sys > 140:
                risk_factors.append("Hypertension")
            if exercise_hours < 2:
                risk_factors.append("Low Exercise")
            
            st.markdown("**Risk Factors:**")
            if risk_factors:
                for rf in risk_factors:
                    st.write(f"⚠️ {rf}")
            else:
                st.write("✅ Low risk profile")
            
            # Patient similarity visualization
            st.markdown("**Similar Patients Found:**")
            st.metric("Total Similar Cases", len(similar_patients))
            
            # Treatment distribution in similar patients
            treatment_dist = similar_patients['treatment'].value_counts()
            fig = px.pie(values=treatment_dist.values, names=treatment_dist.index,
                        title="Treatments Used in Similar Cases")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.markdown('<div class="section-header">📈 Detailed Analysis</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rates comparison
            treatments = [t[0].replace('_', ' ') for t in recommendations]
            success_rates = [t[1]['success_rate'] for t in recommendations]
            
            fig = go.Figure(data=[
                go.Bar(x=treatments, y=success_rates, 
                      marker_color=['#28a745' if x >= 0.7 else '#ffc107' if x >= 0.5 else '#dc3545' 
                                   for x in success_rates])
            ])
            fig.update_layout(title="Treatment Success Rates",
                            yaxis_title="Success Rate",
                            yaxis=dict(range=[0, 1], tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Patient characteristics vs database
            st.markdown("**Your Profile vs Database Average:**")
            
            comparisons = [
                ("Age", age, df['age'].mean()),
                ("BMI", bmi, df['bmi'].mean()),
                ("HbA1c", hba1c, df['hba1c'].mean()),
                ("Glucose", glucose_level, df['glucose_level'].mean())
            ]
            
            for metric, patient_val, avg_val in comparisons:
                diff = ((patient_val - avg_val) / avg_val) * 100
                color = "red" if abs(diff) > 20 else "orange" if abs(diff) > 10 else "green"
                st.metric(
                    label=metric,
                    value=f"{patient_val:.1f}",
                    delta=f"{diff:+.1f}% vs avg",
                    delta_color="inverse" if metric in ["BMI", "HbA1c", "Glucose"] else "normal"
                )
    
    # Database overview
    with st.expander("📊 Treatment Database Overview"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(df))
            st.metric("Success Rate", f"{df['treatment_success'].mean():.1%}")
        
        with col2:
            most_common_treatment = df['treatment'].mode()[0]
            st.metric("Most Common Treatment", most_common_treatment.replace('_', ' '))
            st.metric("Average Age", f"{df['age'].mean():.1f} years")
        
        with col3:
            st.metric("Average HbA1c", f"{df['hba1c'].mean():.1f}%")
            st.metric("Average BMI", f"{df['bmi'].mean():.1f}")
        
        # Treatment outcomes by type
        treatment_outcomes = df.groupby('treatment')['treatment_success'].agg(['count', 'mean']).round(3)
        treatment_outcomes.columns = ['Patient Count', 'Success Rate']
        treatment_outcomes['Success Rate'] = treatment_outcomes['Success Rate'].apply(lambda x: f"{x:.1%}")
        st.dataframe(treatment_outcomes, use_container_width=True)

if __name__ == "__main__":
    main()
