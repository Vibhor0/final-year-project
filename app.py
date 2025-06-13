from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, and_
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from models import db, User, Complaint, Feedback
from flask_migrate import Migrate
from flask_socketio import SocketIO, send
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import json
from flask import Flask, request, jsonify
import joblib  
from flask import g
import pickle
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask_socketio import SocketIO, send
import re

with open('project-model-pickle-files/department-classifier/tfidf_vectorizer.pkl', 'rb') as f:
    tf_vectorizer = pickle.load(f)

with open('project-model-pickle-files/department-classifier/model.pkl', 'rb') as f:
    department_classifier_model = pickle.load(f)

with open('project-model-pickle-files/department-classifier/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


try:
    with open('project-model-pickle-files/urgency-predict/tfidf_vectorizer_current.pkl', 'rb') as f:
        urgency_tfidf_vectorizer = joblib.load(f)
    with open('project-model-pickle-files/urgency-predict/logistic_regression_classifier_current.pkl', 'rb') as f:
        urgency_classifier_model = joblib.load(f)
    print("New urgency classification model components loaded successfully.")
except FileNotFoundError:
    print("Error: New urgency model files not found. Ensure 'tfidf_vectorizer_current.pkl' and 'logistic_regression_classifier_current.pkl' are in 'project-model-pickle-files/urgency-predict/'.")
    urgency_tfidf_vectorizer = None
    urgency_classifier_model = None
except Exception as e:
    print(f"An error occurred while loading new urgency model components: {e}")
    urgency_tfidf_vectorizer = None
    urgency_classifier_model = None

app = Flask(__name__)
socketio = SocketIO(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///railway_grievance.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
socketio = SocketIO(app)

migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_message = None


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

login_manager.login_view = 'login'

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

def analyze_sentiment(feedback_text):
    blob = TextBlob(feedback_text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def generate_feedback_trend_graph(trend_data):
    months = [data[0] for data in trend_data]
    counts = [data[1] for data in trend_data]
    plt.figure(figsize=(10, 5))
    plt.plot(months, counts, marker='o', color='b')
    plt.title("Feedback Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Feedbacks") 
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from datetime import datetime
import numpy as np
import pandas as pd
from textblob import TextBlob
import pickle 

def extract_features(text):
    text_length = len(text)
    num_words = len(text.split())
    sentiment = TextBlob(text).sentiment.polarity
    return np.array([[text_length, num_words, sentiment]])


# -------------------- Urgency Calculation --------------------
def calculate_urgency(complaint):
    if urgency_tfidf_vectorizer is None or urgency_classifier_model is None:
        print("Urgency classification model not loaded. Returning 'Medium' by default.")
        return "Medium" 
    combined_text = f"{complaint.department} {complaint.additional_info}"
    text_features_transformed = urgency_tfidf_vectorizer.transform([combined_text])
    predicted_urgency_label = urgency_classifier_model.predict(text_features_transformed)[0]
    return predicted_urgency_label

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def home():
    return redirect(url_for('user_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role.lower() == 'admin':
                return redirect(url_for('admin_dashboard', department=user.department))
            elif user.role.lower() == 'employee':
                return redirect(url_for('employee_dashboard'))  
            return redirect(url_for('user_dashboard'))  
        else:
            print("hello") 
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])  
        role = request.form['role']
        department = request.form['department'] if role in ['Admin', 'Employee'] else None         
        new_user = User(username=username, password=password, role=role, department=department)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

def determine_department(additional_info):
    transformed_text = tf_vectorizer.transform([additional_info])
    predicted_label = department_classifier_model.predict(transformed_text)
    department = label_encoder.inverse_transform(predicted_label)[0]
    return department

@app.route('/complaint', methods=['GET', 'POST'])
@login_required
def complaint():
    if request.method == 'POST':
        age_str = request.form.get('age')
        pnr_no = request.form.get('pnr_no')
        additional_info = request.form.get('additional_info', '')
        errors = []
        if not pnr_no:
            errors.append('PNR Number is required.')
        elif not pnr_no.isdigit() or len(pnr_no) != 10:
            errors.append('PNR Number must be a 10-digit number.')
        if not age_str:
            errors.append('Age is required.')
        else:
            try:
                age = int(age_str)
                if not (1 <= age <= 120):
                    errors.append('Age must be between 1 and 120.')
            except ValueError:
                errors.append('Invalid age. Please enter a valid number.')
        if errors:
            for error in errors:
                flash(error, 'complaint_error')
            return redirect(url_for('complaint'))
        age = int(age_str)
        unique_id = str(datetime.now().timestamp())
        department = determine_department(additional_info)
        print(department)
        images = request.files.getlist('images')
        image_paths = []
        for image in images:
            if image:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_paths.append(filename)
        images_str = ','.join(image_paths)
        employee = (
            db.session.query(User)
            .filter_by(role='Employee', department=department)
            .outerjoin(Complaint, and_(Complaint.assigned_employee_id == User.id, Complaint.status == 'Unsolved'))
            .group_by(User.id)
            .order_by(func.count(Complaint.id).asc())
            .first()
        )
        complaint = Complaint(
            unique_id=unique_id,
            department=department,
            date=datetime.now().date(),
            time=datetime.now().time(),
            pnr_no=pnr_no,
            age=age,
            additional_info=additional_info,
            images=images_str,
            user_id=current_user.id,
            assigned_employee_id=employee.id if employee else None
        )
        complaint.urgency = calculate_urgency(complaint)
        db.session.add(complaint)
        db.session.commit()
        return redirect(url_for('user_dashboard'))
    return render_template('complaint.html')


@app.route('/track_complaints')
@login_required
def track_complaints():
    complaints = Complaint.query.filter(
        (Complaint.user_id == current_user.id) &
        ((Complaint.status == 'Unsolved') | (~Complaint.feedbacks.any()))
    ).all()
    return render_template('track_complaints.html', complaints=complaints)


@app.route('/submit_feedback/<int:complaint_id>', methods=['GET', 'POST'])
@login_required
def submit_feedback(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    if complaint.status != 'Completed':
        return redirect(url_for('track_complaints'))
    if request.method == 'POST':
        feedback_text = request.form['feedback_text']
        rating = request.form.get('rating', type=int)
        sentiment = analyze_sentiment(feedback_text)
        existing_feedback = Feedback.query.filter_by(complaint_id=complaint.id).first()
        if existing_feedback:
            return redirect(url_for('track_complaints'))
        feedback = Feedback(
            complaint_id=complaint.id,
            feedback_text=feedback_text,
            sentiment=sentiment,
            rating=rating
        )
        db.session.add(feedback)
        complaint.feedback_status = 'Completed'
        db.session.commit()
        return redirect(url_for('user_dashboard'))
    return render_template('feedback_form.html', complaint=complaint)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@app.route('/admin_dashboard/<department>')
@login_required
def admin_dashboard(department):
    complaints = Complaint.query.filter_by(department=department).all()
    for complaint in complaints:
        feedback = Feedback.query.filter_by(complaint_id=complaint.id).first()
        complaint.feedback = feedback
    feedbacks = Feedback.query.join(Complaint).filter(Complaint.department == department).all()
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for feedback in feedbacks:
        if feedback.sentiment in sentiment_counts:
            sentiment_counts[feedback.sentiment] += 1
    ratings = [feedback.rating for feedback in feedbacks if feedback.rating is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else "No ratings yet"
    feedback_dates = [feedback.created_at.date().isoformat() for feedback in feedbacks if feedback.created_at]
    feedback_trend = Counter(feedback_dates)
    sorted_trend_data = sorted(feedback_trend.items())
    trend_data = {
        "labels": [date for date, _ in sorted_trend_data],
        "data": [count for _, count in sorted_trend_data]
    }
    return render_template('admin_dashboard.html', complaints=complaints, sentiment_counts=sentiment_counts, avg_rating=avg_rating, trend_data=json.dumps(trend_data))


@app.route('/admin/employees')
@login_required
def view_employees():
    if current_user.role != 'Admin':
        return redirect(url_for('home'))
    employees = User.query.filter_by(role='Employee', department=current_user.department).all()
    return render_template('view_employees.html', employees=employees, department=current_user.department)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/employee_dashboard')
@login_required
def employee_dashboard():
    if current_user.role != 'Employee':
        return redirect(url_for('home'))
    assigned_complaints = Complaint.query.filter(
        Complaint.assigned_employee_id == current_user.id,
        Complaint.status != 'Completed'
    ).order_by(Complaint.urgency.desc()).all()
    return render_template('employee_dashboard.html', complaints=assigned_complaints)

@app.route('/mark_as_solved/<complaint_id>', methods=['POST'])
@login_required
def mark_as_solved(complaint_id):
    complaint = Complaint.query.filter_by(unique_id=complaint_id).first()
    if complaint:
        complaint.status = 'Completed'
        db.session.commit()
    return redirect(url_for('employee_dashboard'))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
HISTORY_FILE = "conversation_history.json"
prompt1 = ChatPromptTemplate.from_messages([
    ("system", """You are an Indian Railways complaint assistant named Rail Sahayak.
     Your job is to assist users by responding to their complaints in a helpful and reassuring way.
     Please respond with empathy and understanding. Your responses should be clear, concise, and not repetitive.
     Provide logical and actionable solutions in bullet points when necessary.
     Limit your response to 3-4 lines."""),
    ("user", "Complaint: {question}")
])

llm = Ollama(model="tinyllama")
output_parser = StrOutputParser()
chain1 = prompt1 | llm | output_parser
def load_conversation_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as json_file:
                return json.load(json_file)
    except IOError as e:
        print(f"Error loading conversation history: {e}")
    return []

conversation_history = load_conversation_history()

def handle_input(input_text):
    train_keywords = {
        "general": ["train", "railway", "rail", "station", "platform", "route", "coach"],
        "tickets": ["ticket", "pnr", "reservation", "cancel", "refund", "booking", "fare", "ticketless"],
        "seat": ["seat", "berth", "compartment", "coach", "cleaning", "AC", "non-AC"],
        "schedule": ["arrival", "departure", "delay", "timing", "reschedule"],
        "emergency": ["medical", "medicine", "doctor", "first aid", "emergency", "stampede"],
        "luggage": ["luggage", "bag", "lost", "baggage", "theft", "missing"],
        "food": ["food", "meal", "catering", "pantry", "water", "snacks"],
        "safety": ["accident", "theft", "police", "security", "help", "assistance", "overcrowding", "too many people", "so many people"],
        "travel": ["journey", "destination", "boarding", "deboarding", "layover"],
    } 
    all_keywords = [kw for sublist in train_keywords.values() for kw in sublist]
    if not any(keyword in input_text.lower() for keyword in all_keywords):
        return "This is not related to my purpose."
    raw_output = chain1.invoke({'question': input_text})
    response_lines = raw_output.split("\n")
    filtered_response = []  
    for line in response_lines:
        if re.match(r'^\w+:', line) and not line.startswith("AI:"):  
            break
        filtered_response.append(line.strip())
    full_response = " ".join(filtered_response).strip()
    full_response = "AI: " + full_response if not full_response.startswith("AI:") else full_response  
    conversation_history.append({
        "question": input_text,
        "response": full_response
    })
    with open(HISTORY_FILE, "w") as json_file:
        json.dump(conversation_history, json_file, indent=4)
    return full_response

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    if request.method == 'POST':
        user_message = request.form['message']
        bot_response = handle_input(user_message)
        return jsonify(bot_response=bot_response)
    return render_template('chatbot.html')

@socketio.on('message')
def handle_socket_message(msg):
    print('Message from user: ' + msg)
    bot_response = handle_input(msg)
    send(bot_response, broadcast=True)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['message']
    bot_response = handle_input(user_message)
    return jsonify({'user_message': user_message, 'bot_response': bot_response})

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/admin/statistics/<department>')
@login_required
def statistics(department):
    feedbacks = Feedback.query.filter(Complaint.department == department).join(Complaint).all()
    avg_rating = sum(feedback.rating for feedback in feedbacks) / len(feedbacks) if feedbacks else 0
    sentiment_counts = {
        'Positive': sum(1 for feedback in feedbacks if feedback.sentiment == 'Positive'),
        'Negative': sum(1 for feedback in feedbacks if feedback.sentiment == 'Negative'),
        'Neutral': sum(1 for feedback in feedbacks if feedback.sentiment == 'Neutral'),
    }
    trend_data = db.session.query(
        db.func.strftime('%Y-%m', Feedback.timestamp).label('month'),
        db.func.count().label('count')
    ).group_by('month').all()
    trend_graph = generate_feedback_trend_graph(trend_data)
    return render_template('admin_statistics.html', avg_rating=avg_rating, sentiment_counts=sentiment_counts, trend_data=trend_data, trend_graph=trend_graph)

@app.route('/admin/statistics/employee_performance/<department>')
@login_required
def employee_performance(department):
    feedbacks = Feedback.query.join(Complaint).filter(Complaint.department == department).all()
    employee_performance = {}
    for feedback in feedbacks:
        employee = feedback.complaint.assigned_employee
        if employee:
            if employee.id not in employee_performance:
                employee_performance[employee.id] = {'positive': 0, 'negative': 0, 'total': 0}
            employee_performance[employee.id]['total'] += 1
            if feedback.sentiment == 'Positive':
                employee_performance[employee.id]['positive'] += 1
            elif feedback.sentiment == 'Negative':
                employee_performance[employee.id]['negative'] += 1
    return render_template('employee_performance.html', employee_performance=employee_performance)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('user_dashboard'))

@app.route('/view_database')
def view_database():
    users = User.query.all()
    complaints = Complaint.query.all()
    feedbacks = Feedback.query.all()
    feedback_dates = [feedback.created_at.date().isoformat() for feedback in feedbacks if feedback.created_at]
    feedback_trend = Counter(feedback_dates)
    sorted_trend_data = sorted(feedback_trend.items())
    trend_data = {
        "labels": [date for date, _ in sorted_trend_data],
        "data": [count for _, count in sorted_trend_data]
    }
    return render_template('database.html', users=users, complaints=complaints, feedbacks=feedbacks, trend_data=json.dumps(trend_data))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
