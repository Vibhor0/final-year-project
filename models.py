from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime


db = SQLAlchemy()

class User(db.Model, UserMixin):  # Inherit UserMixin for login functionality
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # e.g., 'user', 'admin', 'employee'
    department = db.Column(db.String(100), nullable=True)  # Specify the department for admins
    
    # Relationships to complaints
    complaints = relationship('Complaint', back_populates='user', foreign_keys='Complaint.user_id')
    assigned_complaints = relationship('Complaint', back_populates='assigned_employee', foreign_keys='Complaint.assigned_employee_id')



class Complaint(db.Model):
    __tablename__ = 'complaint'
    id = db.Column(db.Integer, primary_key=True)
    unique_id = db.Column(db.String, unique=True, nullable=False)
    department = db.Column(db.String, nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    pnr_no = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    additional_info = db.Column(db.Text, nullable=True)
    images = db.Column(db.String, nullable=True)
    status = db.Column(db.String(20), default='Unsolved')
    urgency = db.Column(db.Integer, default=0)

    # Foreign key to link Complaint to User (the user who filed the complaint)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    assigned_employee_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    # Relationship to User: the user who filed the complaint
    user = relationship('User', back_populates='complaints', foreign_keys=[user_id])

    # Relationship to assigned_employee (another User)
    assigned_employee = relationship('User', back_populates='assigned_complaints', foreign_keys=[assigned_employee_id])

    # Relationship to Feedback: one-to-many with Feedback (complaints can have multiple feedbacks)
    feedbacks = db.relationship('Feedback', back_populates='complaint', lazy='dynamic')


class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    complaint_id = db.Column(db.Integer, db.ForeignKey('complaint.id'), nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)  # Changed to Text for longer feedback
    status = db.Column(db.String(20), default='Pending')
    sentiment = db.Column(db.String(50), nullable=True)
    rating = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Relationship to Complaint: feedback is associated with a single complaint
    complaint = relationship('Complaint', back_populates='feedbacks')
    def __init__(self, complaint_id, feedback_text, sentiment, rating):
        self.complaint_id = complaint_id
        self.feedback_text = feedback_text
        self.sentiment = sentiment 
        self.rating = rating
