from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn
import logging
import os
import pickle
from pulp import *

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a random secret key in production

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the diabetes prediction model and scaler
model_path = 'models/model.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model successfully loaded. Type: {type(model)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
else:
    print(f"Model file not found at {model_path}")
    model = None

dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

# If model failed to load, train a new one
if model is None:
    print("Creating a new model")
    X = dataset.iloc[:, [1, 2, 5, 7]].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = SVC(kernel='rbf', random_state=0, probability=True)
    model.fit(sc.fit_transform(X_train), y_train)
    print("New model created and trained")

    # Save the new model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print("New model saved")

# Load database datasets for medicine suggestion
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load SVC model for medicine suggestion
try:
    svc = pickle.load(open('models/svc.pkl', 'rb'))
    print("SVC model loaded successfully")
except Exception as e:
    print(f"Error loading SVC model: {str(e)}")

# Constants for diet planning
NUTRITION_CSV = 'Datasets/nutrition.csv'
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
meals = ['Snack 1', 'Snack 2', 'Breakfast', 'Lunch', 'Dinner']
meal_split = {'Snack 1': 0.10, 'Snack 2': 0.10, 'Breakfast': 0.15, 'Lunch': 0.35, 'Dinner': 0.30}
data = None
split_values_meal = None
split_values_day = None

# Database setup
def init_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        height REAL NOT NULL,
                        weight REAL NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL,
                        dob TEXT NOT NULL,
                        glucose REAL,
                        blood_pressure REAL)''')

    conn.commit()
    conn.close()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to log in first!', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    user_id = session.get('user_id')
    if user_id:
        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if user:
            height = user[3]  # Assuming height is at index 3
            weight = user[4]  # Assuming weight is at index 4
            bmi = calculate_bmi(height, weight)
            return render_template('index.html', height=height, weight=weight, bmi=bmi)

    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            email = request.form['email']
            password = request.form['password']
            dob = request.form['dob']

            hashed_password = generate_password_hash(password)

            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (name, age, height, weight, email, password, dob) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, age, height, weight, email, hashed_password, dob)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash('Email already registered!', 'danger')

        except Exception as e:
            print(f"Error during registration: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')

        finally:
            conn.close()

    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[6], password):
            session['user_id'] = user[0]  # Store user ID in session
            session['user_name'] = user[1]  # Store user name in session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash('You have been logged out!', 'success')
    return redirect(url_for('login'))

@app.route('/user')
@login_required
def user():
    user_id = session.get('user_id')
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return render_template('user.html', user=user)

    flash('User not found', 'error')
    return redirect(url_for('index'))

@app.route('/update_user', methods=['POST'])
@login_required
def update_user():
    user_id = session.get('user_id')
    height = float(request.form.get('height'))
    weight = float(request.form.get('weight'))

    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET height=?, weight=? WHERE id=?", (height, weight, user_id))
    conn.commit()
    conn.close()

    flash('User details updated successfully', 'success')
    return redirect(url_for('index'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()

        if user:
            new_password = 'newpassword'  # In a real application, generate a secure random password
            hashed_password = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET password=? WHERE email=?", (hashed_password, email))
            conn.commit()
            flash(f'Password reset successful! Your new password is: {new_password}', 'success')
            # In a real application, you would send this password via email instead of displaying it
        else:
            flash('Email not found!', 'danger')
        conn.close()
    return render_template('forgot_password.html')

@app.route('/diet')
@login_required
def diet():
    return render_template('diet.html')

def safe_predict(model, X):
    try:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            return model.decision_function(X)
        else:
            return model.predict(X)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

@app.route('/diabetes_prediction', methods=['GET', 'POST'])
@login_required
def diabetes_prediction():
    user_id = session.get('user_id')
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    # Retrieve user data
    cursor.execute("SELECT age, height, weight FROM users WHERE id=?", (user_id,))
    user_data = cursor.fetchone()

    if user_data:
        age, height, weight = user_data
        bmi = calculate_bmi(height, weight)
    else:
        age, bmi = None, None

    if request.method == 'POST':
        try:
            # Get the form data
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])

            # Log input data
            logging.info(f"Input data - Glucose: {glucose}, BP: {blood_pressure}, BMI: {bmi}, Age: {age}")

            # Use the data for prediction
            final_features = np.array([[glucose, blood_pressure, bmi, age]])
            logging.info(f"Final features: {final_features}")

            transformed_features = sc.transform(final_features)
            logging.info(f"Transformed features: {transformed_features}")

            # Check if the model is loaded
            if model is None:
                logging.error("Model is not loaded")
                return jsonify({'error': 'Model is not available. Please try again later.'})

            # Make prediction
            prediction = safe_predict(model, transformed_features)
            logging.info(f"Raw prediction: {prediction}")

            if prediction is None:
                return jsonify({'error': 'An error occurred during prediction. Please try again.'})

            # Use a threshold to determine high or low risk
            threshold = 0.3  # Adjust this threshold based on your model's performance
            is_high_risk = prediction[0] > threshold
            logging.info(f"Is high risk: {is_high_risk}")

            pred_text = "You have a high risk of Diabetes, please consult a Doctor." if is_high_risk else "You have a low risk of Diabetes."

            conn.close()
            return jsonify({
                'prediction_text': pred_text,
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'prediction_value': float(prediction[0])  # Add this line to see the actual prediction value
            })
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'})

    # For GET requests, return the template with user's age and BMI
    conn.close()
    return render_template('diabetis.html', age=age, bmi=bmi)

def calculate_bmi(height, weight):
    height_m = float(height) / 100  # Convert cm to m
    bmi = float(weight) / (height_m ** 2)
    return round(bmi, 2)

# Helper function for medicine suggestion
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45,
                 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
                 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52,
                 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57,
                 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
                 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
                 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
                 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
                 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76,
                 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
                 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
                 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list = list(symptoms_dict.keys())

# Model Prediction function for medicine suggestion
def get_predicted_value(patient_symptoms):
    print(f"Symptoms received for prediction: {patient_symptoms}")  # Debug print
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    print(f"Input vector: {input_vector}")  # Debug print
    prediction = svc.predict([input_vector])[0]
    print(f"Raw prediction: {prediction}")  # Debug print
    return diseases_list.get(prediction, "Unknown disease")

@app.route('/medicine_suggestion', methods=['GET', 'POST'])
@login_required
def medicine_suggestion():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        print(f"Received symptoms: {symptoms}")  # Debug print

        if not symptoms:
            message = "Please select at least one symptom"
            return render_template('medicine_suggestion.html', message=message, symptoms_list=symptoms_list)

        try:
            predicted_disease = get_predicted_value(symptoms)
            print(f"Predicted disease: {predicted_disease}")  # Debug print
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('medicine_suggestion.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout, symptoms_list=symptoms_list)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")  # Debug print
            message = "An error occurred during prediction. Please try again."
            return render_template('medicine_suggestion.html', message=message, symptoms_list=symptoms_list)

    return render_template('medicine_suggestion.html', symptoms_list=symptoms_list)

# Function definitions for diet planning
def random_dataset_meal(data):
    global split_values_meal
    frac_data = data.sample(frac=1).reset_index().drop('index', axis=1)
    meal_data = []
    for s in range(len(split_values_meal) - 1):
        meal_data.append(frac_data.loc[split_values_meal[s]:split_values_meal[s + 1]])
    return dict(zip(meals, meal_data))

def random_dataset_day():
    global split_values_day
    frac_data = data.sample(frac=1).reset_index().drop('index', axis=1)
    day_data = []
    for s in range(len(split_values_day) - 1):
        day_data.append(frac_data.loc[split_values_day[s]:split_values_day[s + 1]])
    return dict(zip(week_days, day_data))

def build_nutritional_values(kg, calories):
    protein_calories = kg * 4
    carb_calories = calories / 2.
    fat_calories = calories - carb_calories - protein_calories
    res = {'Protein Calories': protein_calories, 'Carbohydrates Calories': carb_calories, 'Fat Calories': fat_calories}
    return res

def extract_gram(table):
    protein_grams = table['Protein Calories'] / 4.
    carbs_grams = table['Carbohydrates Calories'] / 4.
    fat_grams = table['Fat Calories'] / 9.
    res = {'Protein Grams': protein_grams, 'Carbohydrates Grams': carbs_grams, 'Fat Grams': fat_grams}
    return res

def model(prob, kg, calories, meal, data):
    G = extract_gram(build_nutritional_values(kg, calories))
    E = G['Carbohydrates Grams']
    F = G['Fat Grams']
    P = G['Protein Grams']
    day_data = data
    day_data = day_data[day_data.calories != 0]
    food = day_data.name.tolist()
    c = day_data.calories.tolist()
    x = pulp.LpVariable.dicts("x", indices=food, lowBound=0, upBound=1.5, cat='Continuous', indexStart=[])
    e = day_data.carbohydrate.tolist()
    f = day_data.total_fat.tolist()
    p = day_data.protein.tolist()
    div_meal = meal_split[meal]
    prob += pulp.lpSum([x[food[i]] * c[i] for i in range(len(food))])
    prob += pulp.lpSum([x[food[i]] * e[i] for i in range(len(x))]) >= E * div_meal
    prob += pulp.lpSum([x[food[i]] * f[i] for i in range(len(x))]) >= F * div_meal
    prob += pulp.lpSum([x[food[i]] * p[i] for i in range(len(x))]) >= P * div_meal
    prob.solve()
    variables = []
    values = []
    for v in prob.variables():
        variable = v.name
        value = v.varValue
        variables.append(variable)
        values.append(value)
    values = np.array(values).round(2).astype(float)
    sol = pd.DataFrame(np.array([food, values]).T, columns=['Food', 'Quantity'])
    sol['Quantity'] = sol.Quantity.astype(float)
    sol = sol[sol['Quantity'] != 0.0]
    sol.Quantity = (sol.Quantity * 100).astype(int)
    sol = sol.rename(columns={'Quantity': 'Quantity (g)'})
    return sol

def better_model(kg, calories):
    days_data = random_dataset_day()
    res_model = []
    for day in week_days:
        day_data = days_data[day]
        meals_data = random_dataset_meal(day_data)
        meal_model = []
        for meal in meals:
            meal_data = meals_data[meal]
            prob = pulp.LpProblem("Diet", LpMinimize)
            sol_model = model(prob, kg, calories, meal, meal_data)
            meal_model.append(sol_model.to_dict(orient='records'))
        res_model.append(meal_model)
    unpacked = []
    for i in range(len(res_model)):
        unpacked.append(dict(zip(meals, res_model[i])))
    unpacked_tot = dict(zip(week_days, unpacked))
    return unpacked_tot

def diet_internal(kg, calories):
    global data
    global split_values_meal
    global split_values_day
    # Model load which only happens during cold starts
    if data is None:
        data = pd.read_csv(NUTRITION_CSV).drop('Unnamed: 0', axis=1)
        data = data[['name', 'calories', 'carbohydrate', 'total_fat', 'protein']]
        data['carbohydrate'] = np.array([data['carbohydrate'].tolist()[i].split(' ') for i in range(len(data))])[:, 0].astype('float')
        data['protein'] = np.array([data['protein'].tolist()[i].split(' ') for i in range(len(data))])[:, 0].astype('float')
        data['total_fat'] = np.array([data['total_fat'].tolist()[i].split('g') for i in range(len(data))])[:, 0].astype('float')
        split_values_day = np.linspace(0, len(data), 8).astype(int)
        split_values_day[-1] = split_values_day[-1] - 1
        split_values_meal = np.linspace(0, split_values_day[1], len(meals) + 1).astype(int)
        split_values_meal[-1] = split_values_meal[-1] - 1
    result = better_model(kg, calories)
    return result

@app.route('/diet_plan', methods=['GET', 'POST'])
@login_required
def diet_plan():
    if request.method == 'POST':
        kg = int(request.form['kg'])
        calories = int(request.form['calories'])
        result = diet_internal(kg, calories)
        return render_template('diet.html', kg=kg, calories=calories, result=result)
    return render_template('diet.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)