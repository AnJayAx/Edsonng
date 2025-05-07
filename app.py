from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load the symptom severity data
csv_path = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser/Symptom-severity.csv')
symptom_severity_df = pd.read_csv(csv_path)

# Load the disease precaution data
precaution_path = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser/Disease_precaution.csv')
disease_precaution_df = pd.read_csv(precaution_path)

# Load the machine learning models
models_dir = os.path.join(os.path.dirname(__file__), 'models/Symptom_Analyser')
clf_model = joblib.load(os.path.join(models_dir, 'CLF_model.sav'))
svc_model = joblib.load(os.path.join(models_dir, 'SVC_model.sav'))
xgb_model = joblib.load(os.path.join(models_dir, 'XGB_model.sav'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caduceus-project')
def caduceus_project():
    return render_template('caduceus-project.html')

@app.route('/symptom-analyser', methods=['GET', 'POST'])
def ai_tools():
    if request.method == 'POST':
        # Check if the request is AJAX (JSON) or form submission
        if request.is_json:
            # Get symptoms from JSON data
            data = request.json
            symptoms = data.get('symptoms', [])
            
            # Create a symptom input array with the selected symptoms and padding with zeros
            max_symptoms = 17  # Based on the provided code example
            symptom_input = symptoms + [0] * (max_symptoms - len(symptoms))
            symptom_input = [symptom_input]
            
            # Convert to DataFrame
            user_symptom_df = pd.DataFrame(symptom_input)
            symptom_vals = user_symptom_df.values
            
            print(f"Initial symptoms received: {symptoms}")
            print(f"Symptom input with padding: {symptom_input}")
            print(f"Initial symptom_vals shape: {symptom_vals.shape}")
            print(f"Initial symptom_vals content: {symptom_vals}")
            
            # Replace symptom names with severity weights
            for symptom in symptoms:
                try:
                    weight = symptom_severity_df[symptom_severity_df['Symptom'] == symptom]['weight'].values[0]
                    print(f"Found symptom '{symptom}' with weight: {weight}")
                    symptom_vals[symptom_vals == symptom] = weight
                except (IndexError, KeyError) as e:
                    print(f"Error finding symptom '{symptom}': {e}")
                    # If symptom not found in the severity data, assign default weight
                    symptom_vals[symptom_vals == symptom] = 1
            
            print(f"Symptom_vals after weight assignment: {symptom_vals}")
            
            # Convert any remaining string values to 0
            symptom_vals = np.where(np.char.isdigit(symptom_vals.astype(str)), 
                                   symptom_vals.astype(float), 0)
            
            print(f"Symptom_vals after conversion to numeric: {symptom_vals}")
            
            # Generate a response based on the symptom severity
            total_severity = np.sum(symptom_vals)
            print(f"Total severity score: {total_severity}")
            
            # Use machine learning models to predict disease
            try:
                # Make predictions using all three models
                clf_prediction = clf_model.predict(symptom_vals)[0]
                svc_prediction = svc_model.predict(symptom_vals)[0]
                xgb_prediction = xgb_model.predict(symptom_vals)[0]
                
                print(f"CLF Model Prediction: {clf_prediction}")
                print(f"SVC Model Prediction: {svc_prediction}")
                print(f"XGB Model Prediction: {xgb_prediction}")
                
                # Use a simple voting mechanism to determine the final disease
                predictions = [clf_prediction, svc_prediction, xgb_prediction]
                # Most common prediction (mode)
                from collections import Counter
                prediction_counts = Counter(predictions)
                disease = prediction_counts.most_common(1)[0][0]
                
                # Fallback to severity-based classification if models disagree
                if len(prediction_counts) == 3:  # All models predicted different diseases
                    print("Models disagree, using severity-based fallback")
                    if total_severity > 15:
                        severity_level = "High Severity"
                    elif total_severity > 10:
                        severity_level = "Moderate Severity"
                    else:
                        severity_level = "Mild"
                    disease = f"{disease} ({severity_level})"
            except Exception as e:
                print(f"Error in model prediction: {e}")
                # Fallback to severity-based classification
                if total_severity > 15:
                    disease = "High Severity Condition"
                elif total_severity > 10:
                    disease = "Moderate Severity Condition"
                else:
                    disease = "Mild Condition"
            
            print(f"Final disease prediction: {disease}")
            
            # Get precautions for the predicted disease
            precautions = []
            try:
                # Extract the base disease name without any severity indicators in parentheses
                base_disease = disease.split('(')[0].strip()
                print(f"Looking for precautions for base disease: {base_disease}")
                
                # Find the row with the predicted disease
                disease_row = disease_precaution_df[disease_precaution_df['Disease'].str.contains(base_disease, case=False, na=False)]
                
                # If the disease is found in the precaution dataframe, extract precautions
                if not disease_row.empty:
                    print(f"Found precautions for disease: {disease_row['Disease'].values[0]}")
                    for i in range(1, 5):  # There are 4 precaution columns
                        precaution_col = f'Precaution_{i}'
                        if precaution_col in disease_row.columns:
                            precaution = disease_row.iloc[0][precaution_col]
                            if isinstance(precaution, str) and precaution.strip() and precaution.lower() != 'nan':
                                precautions.append(precaution.strip())
                else:
                    print(f"No specific precautions found for {base_disease}")
                
                if not precautions:
                    precautions = [
                        'Consult with a healthcare professional for specific precautions',
                        'Follow general health guidelines',
                        'Monitor your symptoms closely'
                    ]
            except Exception as e:
                print(f"Error getting precautions: {e}")
                precautions = [
                    'Consult with a healthcare professional for specific precautions',
                    'Follow general health guidelines',
                    'Monitor your symptoms closely'
                ]
            
            print(f"Precautions for {disease}: {precautions}")
            
            response = {
                'disease': disease,
                'description': f'You selected {len(symptoms)} symptoms: {", ".join(symptoms)}',
                'severity_values': symptom_vals.tolist(),
                'total_severity': float(total_severity),
                'recommendations': [
                    'Consult with a healthcare professional',
                    'Monitor your symptoms',
                    'Stay hydrated and get plenty of rest'
                ],
                'precautions': precautions,
                'selected_symptoms': symptoms,
                'model_predictions': {
                    'clf': clf_prediction if 'clf_prediction' in locals() else "N/A",
                    'svc': svc_prediction if 'svc_prediction' in locals() else "N/A",
                    'xgb': xgb_prediction if 'xgb_prediction' in locals() else "N/A"
                }
            }
            return jsonify(response)
        else:
            # Handle regular form submission
            symptoms = request.form.getlist('symptoms[]')
            print(f"Form submission - received symptoms: {symptoms}")
            
            # Create a symptom input array with the selected symptoms and padding with zeros
            max_symptoms = 17  # Based on the provided code example
            symptom_input = symptoms + [0] * (max_symptoms - len(symptoms))
            symptom_input = [symptom_input]
            print(f"Form submission - symptom input with padding: {symptom_input}")
            
            # Convert to DataFrame
            user_symptom_df = pd.DataFrame(symptom_input)
            symptom_vals = user_symptom_df.values
            print(f"Form submission - initial symptom_vals: {symptom_vals}")
            
            # Replace symptom names with severity weights
            for symptom in symptoms:
                try:
                    weight = symptom_severity_df[symptom_severity_df['Symptom'] == symptom]['weight'].values[0]
                    print(f"Found symptom '{symptom}' with weight: {weight}")
                    symptom_vals[symptom_vals == symptom] = weight
                except (IndexError, KeyError) as e:
                    print(f"Error finding symptom '{symptom}': {e}")
                    # If symptom not found in the severity data, assign default weight
                    symptom_vals[symptom_vals == symptom] = 1
            
            print(f"Form submission - symptom_vals after weight assignment: {symptom_vals}")
            
            # Return the results as JSON
            return jsonify({
                'selected_symptoms': symptoms,
                'severity_values': symptom_vals.tolist()
            })
            
    return render_template('symptom-analyser.html')


if __name__ == '__main__':
    app.run(debug=True)
