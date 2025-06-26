import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pandas as pd
import pickle
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'secret_key'

with open('fetal_health_model.pkl', 'rb') as f:
    fetal_health_model = pickle.load(f)
heart_sound_model = load_model('heart_sounds.keras')
# with open('baby_weight_model.pkl', 'rb') as f:
#     baby_weight_model = pickle.load(f)
# with open('features.pkl', 'rb') as f:
#     important_features = pickle.load(f)
# with open('preprocessor.pkl', 'rb') as f:
#     preprocessor = pickle.load(f)
# -------------------------------------------------------------- #

# ------------------ DUMMY HELPERS ------------------ #
def load_data():
    """Load users data from 'users.json'."""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {"patients": [], "doctors": [], "admins": [], "appointments": []}

def save_data(data):
    """Save users data to 'users.json'."""
    with open('users.json', 'w') as f:
        json.dump(data, f)

def fetal_health_classification(data):
    classification = fetal_health_model.predict(data)
    return classification

def heart_sound_classification(file_path, duration=10, sr=22050):
    """
    Classify a heart sound file using the same preprocessing as used for x_test.
    
    Parameters:
      file_path (str): Path to the audio file.
      duration (int): Duration (in seconds) to load from the audio.
      sr (int): Sample rate for loading the audio.
      
    Returns:
      str: Predicted class label (e.g., 'artifact', 'murmur', or 'normal').
    """
    # Load the audio file
    X, sr = librosa.load(file_path, sr=sr, duration=duration)
    
    # Ensure the audio is of the desired duration
    input_length = sr * duration
    if librosa.get_duration(y=X, sr=sr) < duration:
        X = librosa.util.fix_length(data=X, size=input_length)
    
    # Extract MFCC features using 25 coefficients (to match training)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=25).T, axis=0)
    # Reshape to (25, 1) to match the shape of each training sample
    feature = np.array(mfccs).reshape([-1, 1])
    # Add batch dimension: (1, 25, 1)
    input_data = np.expand_dims(feature, axis=0)
    
    # Predict using the trained model
    prediction = heart_sound_model.predict(input_data)
    pred_class = np.argmax(prediction, axis=1)[0]
    return pred_class


def baby_weight_prediction(features):
    """Dummy function returning abnormal (>5) for demonstration."""
    return 10
# ---------------------------------------------------- #


@app.route('/')
def index():
    """Index route: if user is logged in, redirect to their dashboard."""
    if 'user' in session:
        data = load_data()
        for user in data['patients'] + data['doctors'] + data['admins']:
            return redirect(url_for(f"{user['role']}_dashboard"))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route: handle both GET (show form) and POST (attempt login)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        app.logger.info(f"Username {username}")
        data = load_data()
        
        # Check if user exists and password is correct
        for user in data['patients'] + data['doctors'] + data['admins']:
            if user['username'] == username and user['password'] == password:
                session['user'] = user
                app.logger.info(f"Redirecting to {user['role']}")
                if user['role'] == 'admin':
                    return redirect(url_for('admin_dashboard'))
                elif user['role'] == 'doctor':
                    return redirect(url_for('doctor_dashboard'))
                elif user['role'] == 'patient':
                    return redirect(url_for('patient_dashboard'))
                break
        else:
            flash('Invalid credentials', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup route for registering new users."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        data = load_data()

        # Check if username already exists among patients, doctors, or admins
        for user in data['patients'] + data['doctors'] + data['admins']:
            if user['username'] == username:
                flash('Username already exists. Please choose another.', 'danger')
                return redirect(url_for('signup'))
        
        # If not found, create the new user
        user = {"username": username, "password": password, "role": role}
        
        if role == 'patient':
            data['patients'].append(user)
        elif role == 'doctor':
            data['doctors'].append(user)
        else:
            data['admins'].append(user)
        
        save_data(data)
        flash('Account created successfully', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/patient_dashboard')
def patient_dashboard():
    """Patient dashboard after successful login."""
    if 'user' not in session or session['user']['role'] != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_dashboard.html')


@app.route('/admin_dashboard')
def admin_dashboard():
    """Admin dashboard."""
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    data = load_data()
    patients = data.get('patients', [])
    doctors = data.get('doctors', [])
    appointments = data.get('appointments', [])
    return render_template('admin_dashboard.html', 
                           patients=patients, 
                           doctors=doctors, 
                           appointments=appointments)


@app.route('/doctor_dashboard')
def doctor_dashboard():
    """Doctor dashboard showing appointments."""
    if 'user' not in session or session['user']['role'] != 'doctor':
        return redirect(url_for('login'))
    
    data = load_data()
    appointments = [app for app in data['appointments'] if app['doctor'] == session['user']['username']]
    
    return render_template('doctor_dashboard.html', appointments=appointments)



@app.route('/fetal_health_classification', methods=['POST', 'GET'])
def fetal_health():
    """
    Route to handle fetal health classification via CSV:
      - GET -> shows an upload form
      - POST -> processes CSV, calls model for the row
    """
    if request.method == 'POST':
        # Check if a file was submitted
        if 'csv_file' not in request.files:
            flash("No file part in request", "danger")
            return redirect(url_for('fetal_health'))
        
        file = request.files['csv_file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(url_for('fetal_health'))
        
        # Save file to 'uploads' folder
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)  # ensure uploads folder exists
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        try:
            df = pd.read_csv(file_path, index_col=False,header=None)
        except Exception as e:
            flash(f"Error reading CSV file: {e}", "danger")
            return redirect(url_for('fetal_health'))

        # Check if there is at least one row
        if df.empty:
            flash("CSV file is empty.", "danger")
            return redirect(url_for('fetal_health'))
        features = df.T
        features = np.array(features)
        # Process the features through your model
        classification = int(fetal_health_classification(features)[0])
        print(classification) 
        if classification == 1:
            # Normal
            flash('Healthy Fetus', 'success')
            return redirect(url_for('patient_dashboard'))
        elif classification == 2:
            # Suspect
            return render_template('next_steps.html', condition="suspect")
        elif classification == 3:
            # Pathological
            return render_template('next_steps.html', condition="pathological")
    
    return render_template('fetal_health.html')


@app.route('/heart_sound_classification', methods=['POST', 'GET'])
def heart_sound():
    """
    Route to handle heart sound classification:
      - GET -> shows a form for uploading file
      - POST -> processes file, checks classification
    """
    if request.method == 'POST':
        file = request.files['heart_sound_file']
        if file:
            file_path = f'./uploads/{file.filename}'
            file.save(file_path)
            # Dummy abnormal result
            prediction = heart_sound_classification(file_path)  # => 1
            if prediction != 2:  
                reason = "Your heart sound classification was abnormal. Please book an appointment."
                return redirect(url_for('book_appointment', reason=reason))  
            else:
                flash('Healthy Heart Sound', 'success')
                return redirect(url_for('patient_dashboard'))
    return render_template('heart_sound.html')


@app.route('/baby_weight_prediction', methods=['GET', 'POST'])
def baby_weight():
    """
    Route to handle baby weight prediction:
      - GET -> dynamically generates a form based on the feature list,
              including instructions for each feature.
      - POST -> processes the user inputs, uses the model to predict weight,
               and then redirects based on the predicted value.
    """
    # Dictionary mapping feature names to their descriptions
    feature_descriptions = {
        "ID": "Unique Identification number of a baby",
        "SEX": "Sex of the baby",
        "MARITAL": "Marital status of its parents",
        "FAGE": "Age of father",
        "GAINED": "Weight gained during pregnancy",
        "VISITS": "Number of prenatal visits",
        "MAGE": "Age of mother",
        "FEDUC": "Father's years of education",
        "MEDUC": "Mother's years of education",
        "TOTALP": "Total pregnancies",
        "BDEAD": "Number of children born alive now dead",
        "TERMS": "Number of other terminations",
        "LOUTCOME": "Outcome of last delivery",
        "WEEKS": "Completed weeks of gestation",
        "RACEMOM": "Race of mother/child (0: 'Unknown', 1: 'OTHER_NON_WHITE', 2: 'WHITE', 3: 'BLACK', 4: 'AMERICAN_INDIAN', 5: 'CHINESE', 6: 'JAPANESE', 7: 'HAWAIIAN', 8: 'FILIPINO', 9: 'OTHER_ASIAN')",
        "RACEDAD": "Race of Father (0: 'Unknown', 1: 'OTHER_NON_WHITE', 2: 'WHITE', 3: 'BLACK', 4: 'AMERICAN_INDIAN', 5: 'CHINESE', 6: 'JAPANESE', 7: 'HAWAIIAN', 8: 'FILIPINO', 9: 'OTHER_ASIAN')",
        "HISPMOM": "Hispanic status of mother (0: Cubans, 1: Mexicans, 2: No, 3: Colombians, 4: Peruvians, 5: Salvadorans, 6: Guatemalans)",
        "HISPDAD": "Hispanic status of father (0: Cubans, 1: Mexicans, 2: No, 3: Colombians, 4: Peruvians, 5: Salvadorans, 6: Guatemalans)",
        "CIGNUM": "Average number of cigarettes used daily (Mother)",
        "DRINKNUM": "Average number of drinks used daily (Mother)",
        "ANEMIA": "Mother has/had anemia",
        "CARDIAC": "Mother has/had cardiac disease",
        "ACLUNG": "Mother has/had acute or chronic lung disease",
        "DIABETES": "Mother has/had diabetes",
        "HERPES": "Mother has/had genital herpes",
        "HYDRAM": "Mother has/had hydramnios/Oligohydramnios",
        "HEMOGLOB": "Mother has/had hemoglobinopathy",
        "HYPERCH": "Mother has/had chronic hypertension",
        "HYPERPR": "Mother has/had pregnancy hypertension",
        "ECLAMP": "Mother has/had eclampsia",
        "CERVIX": "Mother has/had incompetent cervix",
        "PINFANT": "Mother had/had previous infant 4000+ grams",
        "PRETERM": "Mother has/had previous preterm/small infant",
        "RENAL": "Mother has/had renal disease",
        "RHSEN": "Mother has/had Rh sensitization",
        "UTERINE": "Mother has/had uterine bleeding"
    }

    if request.method == 'GET':
        # Load the feature list from features.pkl
        with open("features.pkl", "rb") as f:
            feature_list = pickle.load(f)
        # Build a dictionary containing descriptions only for the features used by the model.
        feature_desc_to_show = {feature: feature_descriptions.get(feature, "No description available") 
                                for feature in feature_list}
        return render_template('baby_weight.html', feature_list=feature_list, feature_descriptions=feature_desc_to_show)
    
    else:  # POST
        # Load the model from model.pkl
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        # Load the feature list so we know which keys to expect
        with open("features.pkl", "rb") as f:
            feature_list = pickle.load(f)

        # Collect user input for each feature
        user_input = {}
        for feature in feature_list:
            value = request.form.get(feature)
            try:
                user_input[feature] = float(value)
            except Exception as e:
                flash(f"Invalid input for {feature}. Please enter a valid number.", "danger")
                return redirect(url_for('baby_weight'))
        
        # Create a DataFrame from the input (one row)
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        predicted_weight = max(prediction[0], 1.5)
        predicted_weight = round(number=predicted_weight,ndigits=2)
        
        # Dummy condition: if predicted weight > 5, consider it abnormal
        if predicted_weight < 5 or predicted_weight > 8:
            reason = f"Predicted baby weight is {predicted_weight}, indicating a potential issue. Please book an appointment."
            return redirect(url_for('book_appointment', reason=reason))
        else:
            flash(f'Healthy Birth Weight of {predicted_weight}lbs', 'success')
            return redirect(url_for('patient_dashboard'))


@app.route('/book_appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = request.form['patient_name']
        email = request.form['email']  # New field for email
        appointment_date = request.form['appointment_date']
        selected_doctor = request.form['doctor']

        appointment_data = {
            "patient_username": session['user']['username'],
            "patient_name": patient_name,
            "email": email,  # Store the email along with other details
            "appointment_date": appointment_date,
            "doctor": selected_doctor,
            "status": "Pending"
        }

        data = load_data()
        data['appointments'].append(appointment_data)
        save_data(data)
        
        flash('Appointment booked successfully!', 'success')
        return redirect(url_for('patient_dashboard'))
    else:
        # GET -> fetch and display list of doctors + reason
        data = load_data()
        doctors = data.get('doctors', [])
        reason_msg = request.args.get('reason', None)
        return render_template('book_appointment.html', doctors=doctors, reason=reason_msg)


@app.route('/approve_appointment/<int:id>')
def approve_appointment(id):
    """Approve appointment by ID (for doctors only)."""
    if 'user' not in session or session['user']['role'] != 'doctor':
        return redirect(url_for('login'))
    
    data = load_data()
    # Update appointment status to 'Approved'
    data['appointments'][id]['status'] = 'Approved'
    save_data(data)
    
    # Retrieve patient's email from the appointment record (if provided)
    patient_email = data['appointments'][id].get('email')
    if patient_email:
        success, msg = send_verification_email(patient_email, 'Approved')
        if not success:
            flash(f"Appointment approved, but email not sent: {msg}", "warning")
    else:
        flash("Appointment approved, but patient's email is missing.", "warning")
    
    flash('Appointment approved', 'success')
    return redirect(url_for('doctor_dashboard'))

 
def send_verification_email(email, status): 
    try:
        sender_email = "dipenshuqriocity@gmail.com"
        sender_password = "jcgv fags mxtu ttgs"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = 'Appointment Status' 
        
 
        
        body = f"""
        Hello,

        Your appointment has been {status}. 
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)


@app.route('/cancel_appointment/<int:id>')
def cancel_appointment(id):
    """Cancel appointment by ID (for doctors only)."""
    if 'user' not in session or session['user']['role'] != 'doctor':
        return redirect(url_for('login'))
    
    data = load_data()
    # Update appointment status to 'Cancelled'
    data['appointments'][id]['status'] = 'Cancelled'
    save_data(data)
    
    # Retrieve patient's email from the appointment record (if provided)
    patient_email = data['appointments'][id].get('email')
    if patient_email:
        success, msg = send_verification_email(patient_email, 'Cancelled')
        if not success:
            flash(f"Appointment cancelled, but email not sent: {msg}", "warning")
    else:
        flash("Appointment cancelled, but patient's email is missing.", "warning")
    
    flash('Appointment cancelled', 'danger')
    return redirect(url_for('doctor_dashboard'))



@app.route('/logout')
def logout():
    """Clear session and log user out."""
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/view_appointments')
def view_appointments():
    if 'user' not in session or session['user']['role'] != 'patient':
        return redirect(url_for('login'))
    
    data = load_data()
    # Filter appointments for this logged-in patient only (using "patient_username")
    patient_appointments = [
        appt for appt in data['appointments']
        if appt['patient_username'] == session['user']['username']
    ]
    return render_template('view_appointments.html', appointments=patient_appointments)


if __name__ == '__main__':
    app.run(debug=True)