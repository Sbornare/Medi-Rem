from sqlalchemy import UUID
import torch
from flask import Flask, redirect, request, render_template, jsonify, url_for, send_file, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from PIL import Image
import torchvision.transforms as transforms
import io
import re
from paddleocr import PaddleOCR
import numpy as np
import requests
import os
import tempfile
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import uuid
from gtts import gTTS
import traceback

from model import MediRemModel  # Assuming OCRModel is defined elsewhere

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicine_reminder.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Do not use a hardcoded secret key in production
# Generate a random key for development purposes
app.config['SECRET_KEY'] = os.urandom(24).hex()

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Create uploads and reminders directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reminders', exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model
num_classes = 7  # Adjust this number based on your model's requirements
model = MediRemModel(num_classes=num_classes)

# Load the checkpoint (which contains multiple components)
checkpoint = torch.load("best_model.pth")

# Extract just the model state dict from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the appropriate device (CPU or GPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reminders = db.relationship('Reminder', backref='user', lazy=True)
    prescriptions = db.relationship('Prescription', backref='user', lazy=True)

class Reminder(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    medicine = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    frequency = db.Column(db.String(50))
    duration = db.Column(db.String(50))
    method = db.Column(db.String(50))
    reminder_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='upcoming')
    audio_file = db.Column(db.String(100))
    prescription_id = db.Column(db.Integer, db.ForeignKey('prescription.id'), nullable=True)

class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    extracted_text = db.Column(db.Text)
    extracted_medicine_text = db.Column(db.Text)
    reminders = db.relationship('Reminder', backref='prescription', lazy=True)
    
class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    # Default times for medication (stored as hours in 24-hour format)
    once_daily_time = db.Column(db.String(5), default="20:00")  # Default 8:00 PM
    
    twice_daily_time_1 = db.Column(db.String(5), default="08:00")  # Default 8:00 AM
    twice_daily_time_2 = db.Column(db.String(5), default="20:00")  # Default 8:00 PM
    
    thrice_daily_time_1 = db.Column(db.String(5), default="08:00")  # Default 8:00 AM
    thrice_daily_time_2 = db.Column(db.String(5), default="14:00")  # Default 2:00 PM
    thrice_daily_time_3 = db.Column(db.String(5), default="20:00")  # Default 8:00 PM
    
    # Other preferences can be added here in the future
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Regex patterns for medicine details
medicine_pattern = re.compile(r"([a-zA-Z\s]+(?:tab|cap|tablet|capsule))\s*(\d+mg)\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")
syrup_pattern = re.compile(r"([a-zA-Z\s]+(?:syrup))\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")

# Function to generate voice reminder
def generate_voice_reminder(text, reminder_id):
    try:
        # Create reminders directory with absolute path
        reminders_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reminders")
        os.makedirs(reminders_dir, exist_ok=True)
        
        # Create output file with absolute path
        output_file = os.path.join(reminders_dir, f"reminder_{reminder_id}.mp3")
        
        # Generate and save the audio file
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        
        # Return the filename
        return f"reminder_{reminder_id}.mp3"
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# Function to schedule reminders based on extracted medicine data
def schedule_reminders(medicine_data, user_id, prescription_id=None):
    scheduled_reminders = []
    
    user_prefs = UserPreferences.query.filter_by(user_id=user_id).first()
    if not user_prefs:
        # Create default preferences if none exist
        user_prefs = UserPreferences(user_id=user_id)
        db.session.add(user_prefs)
        db.session.commit()
    
    for medicine in medicine_data:
        # Check if medicine is a dictionary (new format) or string (old format)
        if isinstance(medicine, dict):
            name = medicine.get("name", "").strip()
            dosage = medicine.get("dosage", "").strip()
            method = "After Food"  # Default method
            frequency = medicine.get("frequency", "").lower().strip()
            duration = medicine.get("duration", "").lower().strip()
        else:
            # Legacy format - just a string, skip it
            continue

        # Format the medicine information for display
        medicine_display = name
        dosage_display = dosage
        frequency_display = ""
        duration_display = ""

        # Determine the number of times per day and format frequency display
        times_per_day = 1
        if "2x" in frequency or "2 times" in frequency:
            times_per_day = 2
            frequency_display = "2 times a day"
        elif "3x" in frequency or "3 times" in frequency:
            times_per_day = 3
            frequency_display = "3 times a day"
        elif "4x" in frequency or "4 times" in frequency:
            times_per_day = 4
            frequency_display = "4 times a day"
        else:
            frequency_display = "1 time a day"

        # Use user preferences for timing
        reminder_times = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if times_per_day == 1:
            time_str = user_prefs.once_daily_time
            hour, minute = map(int, time_str.split(':'))
            base_time = today.replace(hour=hour, minute=minute)
            reminder_times.append(base_time)
        elif times_per_day == 2:
            time1 = user_prefs.twice_daily_time_1
            time2 = user_prefs.twice_daily_time_2
            
            hour1, minute1 = map(int, time1.split(':'))
            hour2, minute2 = map(int, time2.split(':'))
            
            reminder_times.append(today.replace(hour=hour1, minute=minute1))
            reminder_times.append(today.replace(hour=hour2, minute=minute2))
        elif times_per_day == 3:
            time1 = user_prefs.thrice_daily_time_1
            time2 = user_prefs.thrice_daily_time_2
            time3 = user_prefs.thrice_daily_time_3
            
            hour1, minute1 = map(int, time1.split(':'))
            hour2, minute2 = map(int, time2.split(':'))
            hour3, minute3 = map(int, time3.split(':'))
            
            reminder_times.append(today.replace(hour=hour1, minute=minute1))
            reminder_times.append(today.replace(hour=hour2, minute=minute2))
            reminder_times.append(today.replace(hour=hour3, minute=minute3))
        elif times_per_day == 4:
            # For 4 times, we'll evenly distribute across the day (default behavior)
            # You can add 4-times preferences in the future if needed
            for i in range(times_per_day):
                hour = 8 + (i * 4)  # 8am, 12pm, 4pm, 8pm
                reminder_times.append(today.replace(hour=hour, minute=0))

        # Set start time (current time) and interval
        start_time = datetime.now()
        interval_hours = 24 // times_per_day

        # Determine duration in days and format duration display
        duration_days = 1
        if "week" in duration:
            try:
                weeks = int(duration.split()[0])
                duration_days = weeks * 7
                duration_display = f"{weeks} {'week' if weeks == 1 else 'weeks'}"
            except:
                duration_display = "7 days"
        elif "day" in duration:
            try:
                duration_days = int(duration.split()[0])
                duration_display = f"{duration_days} {'day' if duration_days == 1 else 'days'}"
            except:
                duration_display = "1 day"
        else:
            duration_display = f"{duration_days} day"

        end_time = start_time + timedelta(days=duration_days)

        # Schedule reminders
        for day in range(duration_days):
            for base_time in reminder_times:
                reminder_time = base_time + timedelta(days=day)
                if reminder_time > datetime.now() and reminder_time < end_time:
                    reminder_id = str(uuid.uuid4())
                    reminder_text = f"It's time to take {dosage_display} of {medicine_display} {method}"
                    
                    # Generate voice reminder
                    audio_file = generate_voice_reminder(reminder_text, reminder_id)
                    
                    # Create reminder in database
                    reminder = Reminder(
                        id=reminder_id,
                        user_id=user_id,
                        medicine=medicine_display,
                        dosage=dosage_display,
                        frequency=frequency_display,
                        duration=duration_display,
                        method=method,
                        reminder_time=reminder_time,
                        status="upcoming",
                        audio_file=audio_file,
                        prescription_id=prescription_id
                    )
                    
                    # Add to database
                    db.session.add(reminder)
                    db.session.commit()
                    
                    # Add to scheduled reminders list
                    scheduled_reminders.append(reminder)
                    
                    # Schedule job
                    scheduler.add_job(
                        func=lambda r=reminder: trigger_reminder(r.id),
                        trigger="date",
                        run_date=reminder_time,
                        id=reminder_id
                    )
    
    return scheduled_reminders



# Function to trigger reminder when time is hit
def trigger_reminder(reminder_id):
    with app.app_context():
        reminder = Reminder.query.get(reminder_id)
        if reminder:
            reminder.status = "active"
            db.session.commit()
            print(f"REMINDER ALERT: Take {reminder.medicine} at {reminder.reminder_time}")
    
    return reminder_id

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Home page route redirects to login if not authenticated
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    # Get upcoming reminders for current user
    upcoming_reminders = Reminder.query.filter_by(
        user_id=current_user.id, 
        status='upcoming'
    ).order_by(Reminder.reminder_time).all()
    
    # Get active reminders
    active_reminders = Reminder.query.filter_by(
        user_id=current_user.id, 
        status='active'
    ).order_by(Reminder.reminder_time).all()
    
    return render_template(
        'dashboard.html', 
        upcoming_reminders=upcoming_reminders,
        active_reminders=active_reminders
    )

# Reminders dashboard route
@app.route('/reminders')
@login_required
def view_reminders():
    # Get all reminders for current user
    all_reminders = Reminder.query.filter_by(
        user_id=current_user.id
    ).order_by(Reminder.reminder_time).all()
    
    # Make sure we're getting fresh data from the database, not cached
    db.session.commit()
    
    return render_template('reminders.html', reminders=all_reminders)

@app.route('/get_reminder_details/<reminder_id>')
@login_required
def get_reminder_details(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    
    # Check if reminder belongs to current user
    if reminder.user_id != current_user.id:
        return jsonify({"error": "Unauthorized access"}), 403
    
    return jsonify({
        "id": reminder.id,
        "medicine": reminder.medicine,
        "dosage": reminder.dosage,
        "method": reminder.method,
        "reminder_time": reminder.reminder_time.strftime("%Y-%m-%dT%H:%M"),
        "frequency": reminder.frequency,
        "duration": reminder.duration,
        "status": reminder.status
    })
    

@app.route('/update_reminder/<reminder_id>', methods=['POST'])
@login_required
def update_reminder(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    
    # Check if reminder belongs to current user
    if reminder.user_id != current_user.id:
        return jsonify({"success": False, "error": "Unauthorized access"}), 403
    
    try:
        # Get form data from request
        data = request.get_json() or request.form
        
        # Update reminder fields
        if 'medicine_name' in data:
            reminder.medicine = data['medicine_name'].strip()
        if 'dosage' in data:
            reminder.dosage = data['dosage'].strip()
        if 'method' in data:
            reminder.method = data['method']
        if 'reminder_time' in data:
            reminder.reminder_time = datetime.strptime(data['reminder_time'], "%Y-%m-%dT%H:%M")
            
            # Update scheduler
            try:
                scheduler.remove_job(reminder.id)
            except:
                pass
                
            if reminder.reminder_time > datetime.now():
                scheduler.add_job(
                    func=lambda r=reminder.id: trigger_reminder(r),
                    trigger="date",
                    run_date=reminder.reminder_time,
                    id=reminder.id
                )
                reminder.status = "upcoming"
        
        # Update reminder text and audio
        reminder_text = f"It's time to take {reminder.dosage} of {reminder.medicine} {reminder.method}"
        if reminder.audio_file and os.path.exists(os.path.join('reminders', reminder.audio_file)):
            os.remove(os.path.join('reminders', reminder.audio_file))
        audio_file = generate_voice_reminder(reminder_text, reminder.id)
        reminder.audio_file = audio_file
        
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
    
@app.route('/profile/settings', methods=['GET', 'POST'])
@login_required
def user_settings():
    # Get or create user preferences
    user_prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
    if not user_prefs:
        user_prefs = UserPreferences(user_id=current_user.id)
        db.session.add(user_prefs)
        db.session.commit()
    
    if request.method == 'POST':
        try:
            # Update medication timing preferences
            user_prefs.once_daily_time = request.form.get('once_daily_time')
            user_prefs.twice_daily_time_1 = request.form.get('twice_daily_time_1')
            user_prefs.twice_daily_time_2 = request.form.get('twice_daily_time_2')
            user_prefs.thrice_daily_time_1 = request.form.get('thrice_daily_time_1')
            user_prefs.thrice_daily_time_2 = request.form.get('thrice_daily_time_2')
            user_prefs.thrice_daily_time_3 = request.form.get('thrice_daily_time_3')
            
            db.session.commit()
            flash('Settings updated successfully')
            
            # If the user chose to update existing reminders
            if request.form.get('update_existing') == 'yes':
                update_existing_reminders(current_user.id)
                flash('Existing reminders have been updated with new timing preferences')
                
            return redirect(url_for('user_settings'))
            
        except Exception as e:
            flash(f'Error updating settings: {str(e)}')
    
    return render_template('user_settings.html', user_prefs=user_prefs)

def update_existing_reminders(user_id):
    """Update the timing of existing upcoming reminders based on user preferences"""
    user_prefs = UserPreferences.query.filter_by(user_id=user_id).first()
    if not user_prefs:
        return
    
    # Get all upcoming reminders for the user
    upcoming_reminders = Reminder.query.filter_by(
        user_id=user_id,
        status='upcoming'
    ).all()
    
    for reminder in upcoming_reminders:
        try:
            # Determine frequency to know which time settings to apply
            if "1 time" in reminder.frequency:
                # Get current time components
                current_hour, current_minute = reminder.reminder_time.hour, reminder.reminder_time.minute
                
                # Get preferred time components
                pref_hour, pref_minute = map(int, user_prefs.once_daily_time.split(':'))
                
                # Create a new datetime with adjusted time
                new_time = reminder.reminder_time.replace(hour=pref_hour, minute=pref_minute)
                
            elif "2 times" in reminder.frequency:
                # Determine if this is the first or second reminder of the day
                reminder_hour = reminder.reminder_time.hour
                
                # Get preferred times
                time1_h, time1_m = map(int, user_prefs.twice_daily_time_1.split(':'))
                time2_h, time2_m = map(int, user_prefs.twice_daily_time_2.split(':'))
                
                # Use the time closest to the current reminder time
                if abs(reminder_hour - time1_h) <= abs(reminder_hour - time2_h):
                    new_time = reminder.reminder_time.replace(hour=time1_h, minute=time1_m)
                else:
                    new_time = reminder.reminder_time.replace(hour=time2_h, minute=time2_m)
                
            elif "3 times" in reminder.frequency:
                # Determine which of the three daily reminders this is
                reminder_hour = reminder.reminder_time.hour
                
                # Get preferred times
                time1_h, time1_m = map(int, user_prefs.thrice_daily_time_1.split(':'))
                time2_h, time2_m = map(int, user_prefs.thrice_daily_time_2.split(':'))
                time3_h, time3_m = map(int, user_prefs.thrice_daily_time_3.split(':'))
                
                # Find the closest time
                times = [(time1_h, time1_m), (time2_h, time2_m), (time3_h, time3_m)]
                closest = min(times, key=lambda t: abs(reminder_hour - t[0]))
                
                new_time = reminder.reminder_time.replace(hour=closest[0], minute=closest[1])
            else:
                # Skip reminders with unrecognized frequency
                continue
            
            # Update reminder time if it's in the future
            if new_time > datetime.now():
                # Remove existing job
                try:
                    scheduler.remove_job(reminder.id)
                except:
                    pass
                
                # Update reminder time
                reminder.reminder_time = new_time
                
                # Reschedule job
                scheduler.add_job(
                    func=lambda r=reminder.id: trigger_reminder(r),
                    trigger="date",
                    run_date=new_time,
                    id=reminder.id
                )
        except Exception as e:
            print(f"Error updating reminder {reminder.id}: {e}")
    
    db.session.commit()


@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        
        # Validate username is not taken by another user
        existing_user = User.query.filter(User.username == username, User.id != current_user.id).first()
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('user_settings'))
        
        # Validate email is not taken by another user
        existing_email = User.query.filter(User.email == email, User.id != current_user.id).first()
        if existing_email:
            flash('Email already registered')
            return redirect(url_for('user_settings'))
        
        # Update user
        current_user.username = username
        current_user.email = email
        
        db.session.commit()
        flash('Profile updated successfully')
    
    return redirect(url_for('user_settings'))
# History page route
@app.route('/history')
@login_required
def prescription_history():
    # Get all prescriptions for current user
    prescriptions = Prescription.query.filter_by(
        user_id=current_user.id
    ).order_by(Prescription.upload_date.desc()).all()
    
    return render_template('history.html', prescriptions=prescriptions)

# View specific prescription
@app.route('/prescription/<int:prescription_id>')
@login_required
def view_prescription(prescription_id):
    prescription = Prescription.query.get_or_404(prescription_id)
    
    # Check if prescription belongs to current user
    if prescription.user_id != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('prescription_history'))
    
    # Get reminders related to this prescription
    reminders = Reminder.query.filter_by(
        prescription_id=prescription.id
    ).order_by(Reminder.reminder_time).all()
    
    return render_template(
        'prescription_detail.html',
        prescription=prescription,
        reminders=reminders
    )

# Add custom reminder route
@app.route('/add_reminder', methods=['GET', 'POST'])
@login_required
def add_reminder():
    if request.method == 'POST':
        try:
            medicine_name = request.form.get('medicine_name').strip()
            dosage = request.form.get('dosage').strip()
            method = request.form.get('method')
            reminder_time_str = request.form.get('reminder_time')
            frequency = request.form.get('frequency')
            duration_days = int(request.form.get('duration'))
            
            # Parse reminder time
            reminder_time = datetime.strptime(reminder_time_str, "%Y-%m-%dT%H:%M")
            
            # Format frequency display
            frequency_display = "1 time a day"
            times_per_day = 1
            if frequency == "twice_daily":
                times_per_day = 2
                frequency_display = "2 times a day"
            elif frequency == "thrice_daily":
                times_per_day = 3
                frequency_display = "3 times a day"
            elif frequency == "daily":
                frequency_display = "1 time a day"
            
            # Format duration display
            duration_display = f"{duration_days} {'day' if duration_days == 1 else 'days'}"
            
            interval_hours = 24 // times_per_day
            
            # Create custom reminders
            for day in range(duration_days):
                for i in range(times_per_day):
                    if i == 0 and day == 0:
                        # First reminder uses the exact time specified by user
                        current_reminder_time = reminder_time
                    else:
                        # Calculate time for subsequent reminders
                        hours_to_add = (i * interval_hours) % 24
                        days_to_add = day + (i * interval_hours) // 24
                        current_reminder_time = reminder_time + timedelta(days=days_to_add, hours=hours_to_add)
                    
                    if current_reminder_time > datetime.now():
                        reminder_id = str(uuid.uuid4())
                        reminder_text = f"It's time to take {dosage} of {medicine_name} {method}"
                        
                        # Generate voice reminder
                        audio_file = generate_voice_reminder(reminder_text, reminder_id)
                        
                        # Create new reminder
                        new_reminder = Reminder(
                            id=reminder_id,
                            user_id=current_user.id,
                            medicine=medicine_name,
                            dosage=dosage,
                            frequency=frequency_display,
                            duration=duration_display,
                            method=method,
                            reminder_time=current_reminder_time,
                            status="upcoming",
                            audio_file=audio_file
                        )
                        
                        db.session.add(new_reminder)
                        
                        # Schedule job
                        scheduler.add_job(
                            func=lambda r=reminder_id: trigger_reminder(r),
                            trigger="date",
                            run_date=current_reminder_time,
                            id=reminder_id
                        )
            
            db.session.commit()
            flash('Reminders added successfully')
            return redirect(url_for('view_reminders'))
            
        except Exception as e:
            flash(f'Error adding reminder: {str(e)}')
            return redirect(url_for('add_reminder'))
    
    return render_template('add_reminder.html')

# Route to serve audio files from the reminders directory
@app.route('/reminders/<path:filename>')
@login_required
def serve_audio(filename):
    return send_file(os.path.join('reminders', filename))

# Route to check for active reminders
@app.route('/check_active_reminders')
@login_required
def check_active_reminders():
    current_time = datetime.now()
    
    # Get active reminders for current user
    active_reminders = Reminder.query.filter_by(
        user_id=current_user.id,
        status='active'
    ).all()
    
    # Filter reminders to only include those triggered within the last minute
    recent_active_reminders = [
        {
            'id': reminder.id,
            'medicine': reminder.medicine,
            'dosage': reminder.dosage,
            'method': reminder.method,
            'time': reminder.reminder_time.strftime("%Y-%m-%d %H:%M"),
            'audio_file': reminder.audio_file
        }
        for reminder in active_reminders
        if (current_time - reminder.reminder_time).total_seconds() < 60
    ]
    
    return jsonify({"active_reminders": recent_active_reminders})

# Delete reminder route
@app.route('/delete_reminder/<reminder_id>', methods=['POST'])
@login_required
def delete_reminder(reminder_id):
    try:
        reminder = Reminder.query.get(reminder_id)
        
        # Verify reminder belongs to current user
        if not reminder or reminder.user_id != current_user.id:
            return jsonify({"success": False, "error": "Reminder not found or unauthorized"}), 404
        
        # Try to remove the scheduled job
        try:
            scheduler.remove_job(reminder_id)
        except:
            pass
        
        # Remove the reminder from the database
        db.session.delete(reminder)
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
    
# Add this route to edit reminders
@app.route('/edit_reminder/<reminder_id>', methods=['GET', 'POST'])
@login_required
def edit_reminder(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    
    # Check if reminder belongs to current user
    if reminder.user_id != current_user.id:
        flash('Unauthorized access')
        return redirect(url_for('view_reminders'))
    
    if request.method == 'POST':
        try:
            # Detect JSON or form-data
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            # Update reminder fields
            if data.get('medicine_name'):
                reminder.medicine = data.get('medicine_name').strip()
            if data.get('dosage'):
                reminder.dosage = data.get('dosage').strip()
            if data.get('method'):
                reminder.method = data.get('method')
            if data.get('reminder_time'):
                reminder.reminder_time = datetime.strptime(data.get('reminder_time'), "%Y-%m-%dT%H:%M")
                
                # Update scheduler
                try:
                    scheduler.remove_job(reminder.id)
                except:
                    pass
                    
                if reminder.reminder_time > datetime.now():
                    scheduler.add_job(
                        func=lambda r=reminder.id: trigger_reminder(r),
                        trigger="date",
                        run_date=reminder.reminder_time,
                        id=reminder.id
                    )
                    reminder.status = "upcoming"
            
            # Update reminder text and audio
            reminder_text = f"It's time to take {reminder.dosage} of {reminder.medicine} {reminder.method}"
            if reminder.audio_file and os.path.exists(os.path.join('reminders', reminder.audio_file)):
                os.remove(os.path.join('reminders', reminder.audio_file))
            audio_file = generate_voice_reminder(reminder_text, reminder.id)
            reminder.audio_file = audio_file
            
            db.session.commit()
            
            if request.is_json:
                return jsonify({"success": True})
            else:
                flash('Reminder updated successfully')
                return redirect(url_for('view_reminders'))
                
        except Exception as e:
            if request.is_json:
                return jsonify({"success": False, "error": str(e)}), 400
            else:
                flash(f'Error updating reminder: {str(e)}')
                return redirect(url_for('edit_reminder', reminder_id=reminder_id))
    
    # GET request - show the edit form
    return render_template('edit_reminder.html', reminder=reminder)



# Improve the scheduler initialization
def init_scheduler():
    """Initialize the scheduler and restore any missed reminders"""
    if not scheduler.running:
        scheduler.start()
    
    # Re-schedule upcoming reminders that might have been lost on restart
    with app.app_context():
        upcoming_reminders = Reminder.query.filter(
            Reminder.reminder_time > datetime.now(),
            Reminder.status == 'upcoming'
        ).all()
        
        for reminder in upcoming_reminders:
            try:
                scheduler.add_job(
                    func=lambda r=reminder.id: trigger_reminder(r),
                    trigger="date",
                    run_date=reminder.reminder_time,
                    id=reminder.id
                )
                print(f"Restored reminder: {reminder.id} for {reminder.medicine} at {reminder.reminder_time}")
            except Exception as e:
                print(f"Error restoring reminder {reminder.id}: {str(e)}")

# Mark reminder as taken
@app.route('/mark_taken/<reminder_id>', methods=['POST'])
@login_required
def mark_reminder_taken(reminder_id):
    try:
        reminder = Reminder.query.get(reminder_id)
        
        # Verify reminder belongs to current user
        if not reminder or reminder.user_id != current_user.id:
            return jsonify({"success": False, "error": "Reminder not found or unauthorized"}), 404
        
        # Update reminder status
        reminder.status = "taken"
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# Upload image and run inference
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        image = None  # Placeholder for the image data
        filename = None

        # Check if an image file is uploaded
        if 'file' in request.files and request.files['file']:
            uploaded_file = request.files['file']
            
            # Save the uploaded file
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uploaded_file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            
            image = Image.open(file_path).convert("RGB")

        # Check if an image URL is provided
        if image is None:
            image_url = request.form.get('image_url') or (request.get_json(silent=True) or {}).get('image_url')

            if image_url:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()

                # Validate that the response contains an image
                if 'image' not in response.headers.get('Content-Type', ''):
                    return jsonify({"error": "The provided URL does not point to an image"})

                # Save the image from the URL
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-from-url.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Load the image from the response
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Ensure an image was successfully loaded
        if image is None:
            return jsonify({"error": "No valid image file or URL provided."})

        # Convert image to NumPy array
        image_np = np.array(image)

        # Perform OCR using PaddleOCR
        ocr_results = ocr_engine.ocr(image_np, cls=True)

        # Ensure OCR results are valid
        if not ocr_results or not ocr_results[0]:
            return jsonify({"error": "No text detected in the image."})

        # Extract text from OCR results
        extracted_text = "\n".join([line[1][0] for line in ocr_results[0]])

        # Process extracted text for medicine-related data
        extracted_medicine_data = extract_medicine_data(extracted_text)
        
        # Create a new prescription record
        new_prescription = Prescription(
            user_id=current_user.id,
            filename=filename,
            extracted_text=extracted_text
        )
        
        db.session.add(new_prescription)
        db.session.commit()
        
        # Schedule reminders based on the extracted medicine data
        scheduled_reminders = []
        try:
            if extracted_medicine_data:
                scheduled_reminders = schedule_reminders(
                    extracted_medicine_data, 
                    current_user.id,
                    new_prescription.id
                )
        except Exception as e:
            print(f"Error scheduling reminders: {e}")

        return jsonify({
            "success": True,
            "prescription_id": new_prescription.id,
            "extracted_text": extracted_text,
            "extracted_medicine_data": extracted_medicine_data,
            "reminders_scheduled": len(scheduled_reminders)
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching image from URL: {str(e)}"})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})
    
       
def extract_medicine_data(text):
    # Process the extracted text to identify complete medicine entries
    # Each medicine entry should include name, dosage, frequency, and duration
    medicine_data = []
    
    # Define regex patterns for medicine components
    import re
    medicine_name_pattern = re.compile(r'(Tab|TAB|tab|Cap|CAP|cap|Syrup|SYRUP|syrup|SYP|syp|Syp|Syr|syr|SYR)\s+([A-Za-z0-9\s]+)')
    dosage_pattern = re.compile(r'(\d+)\s*(mg|g|ml|mcg)?')
    frequency_pattern = re.compile(r'(\d+)\s*(times a day|times|x)')
    duration_pattern = re.compile(r'(\d+)\s*(days|day|weeks|week)')
    
    # Process text line by line
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        medicine_match = medicine_name_pattern.search(line)
        
        if medicine_match:
            # Found a medicine name
            medicine_type = medicine_match.group(1)  # Tab, Cap, Syrup, etc.
            medicine_name = medicine_match.group(2).strip()  # The actual name
            full_medicine_name = f"{medicine_type} {medicine_name}"
            
            # Initialize medicine entry
            medicine_entry = {
                "name": full_medicine_name,
                "dosage": "",
                "frequency": "",
                "duration": ""
            }
            
            # Look for dosage in current line or next line
            dosage_match = dosage_pattern.search(line)
            if not dosage_match and i+1 < len(lines):
                dosage_match = dosage_pattern.search(lines[i+1])
                if dosage_match:
                    i += 1  # Move to next line since we found dosage there
            
            if dosage_match:
                dosage_value = dosage_match.group(1)
                dosage_unit = dosage_match.group(2) if dosage_match.group(2) else "mg"
                medicine_entry["dosage"] = f"{dosage_value}{dosage_unit}"
            
            # Look for frequency in current line or next lines
            frequency_match = frequency_pattern.search(line)
            if not frequency_match and i+1 < len(lines):
                frequency_match = frequency_pattern.search(lines[i+1])
                if frequency_match:
                    i += 1  # Move to next line
            
            if frequency_match:
                frequency_value = frequency_match.group(1)
                medicine_entry["frequency"] = f"{frequency_value}x"
            else:
                medicine_entry["frequency"] = "1x"  # Default frequency
            
            # Look for duration in current line or next lines
            duration_match = duration_pattern.search(line)
            if not duration_match and i+1 < len(lines):
                duration_match = duration_pattern.search(lines[i+1])
                if duration_match:
                    i += 1  # Move to next line
            
            if duration_match:
                duration_value = duration_match.group(1)
                duration_unit = duration_match.group(2)
                medicine_entry["duration"] = f"{duration_value} {duration_unit}"
            else:
                medicine_entry["duration"] = "7 days"  # Default duration
            
            # Add complete medicine entry to the list
            medicine_data.append(medicine_entry)
        
        i += 1  # Move to next line

    return medicine_data

# Create database tables before first request
@app.before_request
def create_tables():
    db.create_all()

# Custom filter for formatting dates in templates
@app.template_filter('format_datetime')
def format_datetime(value, format='%Y-%m-%d %H:%M'):
    if value is None:
        return ""
    return value.strftime(format)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(host='0.0.0.0', debug=True)