from flask import Flask, redirect, request, render_template, jsonify, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from PIL import Image
import io
import re
import numpy as np
import requests
import os
import tempfile
from datetime import datetime, timedelta
import uuid
import traceback

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicine_reminder.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Generate a random key for production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simple medicine database for Vercel deployment
MEDICINE_DATABASE = {
    "antibiotics": ["amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline"],
    "painkillers": ["paracetamol", "ibuprofen", "aspirin", "diclofenac"],
    "antacids": ["omeprazole", "pantoprazole", "ranitidine"],
    "vitamins": ["vitamin d", "vitamin b12", "folic acid", "iron"],
    "diabetes": ["metformin", "insulin", "glimepiride"],
    "hypertension": ["amlodipine", "losartan", "atenolol"]
}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    medicines = db.relationship('Medicine', backref='user', lazy=True, cascade='all, delete-orphan')
    reminders = db.relationship('Reminder', backref='user', lazy=True, cascade='all, delete-orphan')

class Medicine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    duration = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medicine_id = db.Column(db.Integer, db.ForeignKey('medicine.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reminder_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, completed, missed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    medicine = db.relationship('Medicine', backref='reminder_instances')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def extract_medicines_simple(text):
    """Simple medicine extraction for Vercel deployment"""
    found_medicines = []
    text_lower = text.lower()
    
    for category, medicines in MEDICINE_DATABASE.items():
        for medicine in medicines:
            if medicine in text_lower:
                # Simple dosage extraction
                dosage_pattern = r'(\d+(?:\.\d+)?)\s*(?:mg|ml|g|tablets?|caps?)'
                dosage_match = re.search(dosage_pattern, text_lower, re.IGNORECASE)
                dosage = dosage_match.group(0) if dosage_match else "As prescribed"
                
                found_medicines.append({
                    'name': medicine.title(),
                    'dosage': dosage,
                    'frequency': 'As prescribed',
                    'duration': 'As prescribed',
                    'method': 'After food'
                })
    
    return found_medicines

def simple_ocr(image_path):
    """Placeholder OCR function for Vercel deployment"""
    # In a real Vercel deployment, you might use a cloud OCR service
    # For now, return a sample text that includes common medicine names
    return "Sample prescription text containing Paracetamol 500mg twice daily for 5 days"

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form.get('phone_number', '')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return render_template('register.html')
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            phone_number=phone_number
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
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

@app.route('/dashboard')
@login_required
def dashboard():
    user_medicines = Medicine.query.filter_by(user_id=current_user.id).order_by(Medicine.created_at.desc()).all()
    active_reminders = Reminder.query.filter_by(user_id=current_user.id, status='pending').order_by(Reminder.reminder_time).all()
    
    return render_template('dashboard.html', 
                         medicines=user_medicines, 
                         reminders=active_reminders)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Simple OCR processing (placeholder for Vercel)
        extracted_text = simple_ocr(file_path)
        
        # Extract medicines
        medicines = extract_medicines_simple(extracted_text)
        
        # If no medicines found, provide fallback
        if not medicines:
            medicines = [{
                'name': 'Medicine detected from prescription',
                'dosage': 'As prescribed',
                'frequency': 'As prescribed',
                'duration': 'As prescribed',
                'method': 'As prescribed'
            }]

        return jsonify({
            'extracted_text': extracted_text,
            'medicines': medicines,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/add_medicine', methods=['POST'])
@login_required
def add_medicine():
    try:
        data = request.get_json()
        
        medicine = Medicine(
            name=data['name'],
            dosage=data['dosage'],
            frequency=data['frequency'],
            duration=data['duration'],
            notes=data.get('notes', ''),
            user_id=current_user.id
        )
        
        db.session.add(medicine)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Medicine added successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/user_settings')
@login_required
def user_settings():
    return render_template('user_settings.html')

@app.route('/test_whatsapp', methods=['POST'])
@login_required
def test_whatsapp():
    try:
        phone_number = request.json.get('phone_number')
        if not phone_number:
            return jsonify({'success': False, 'error': 'Phone number required'})
        
        # For Vercel deployment, we'll simulate WhatsApp sending
        # In production, you'd integrate with WhatsApp Business API
        return jsonify({
            'success': True, 
            'message': 'WhatsApp test simulation completed (Vercel deployment)'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Initialize database
with app.app_context():
    db.create_all()

# Vercel expects the app object to be available
if __name__ == '__main__':
    app.run(debug=False)