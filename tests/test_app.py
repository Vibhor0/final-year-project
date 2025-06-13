import pytest
from app import app, db, User, Complaint, Feedback # Make sure Feedback is imported
from werkzeug.security import generate_password_hash, check_password_hash # For creating/checking test user password
from flask_login import current_user
from datetime import datetime

# Configure app for testing
@pytest.fixture(scope='module')
def test_client():
    # Set app config for testing
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:' # Use in-memory SQLite for tests
    app.config['SECRET_KEY'] = 'a_super_secret_test_key_for_session' # Crucial for session (flash messages)

    with app.test_client() as client:
        with app.app_context():
            db.create_all() # Create tables for the in-memory db

            # Create a test user and log them in
            hashed_password = generate_password_hash('testpassword', method='pbkdf2:sha256')
            test_user = User(username='testuser', password=hashed_password, role='user', department='General')
            db.session.add(test_user)
            db.session.commit()

            # Log in the test user
            login_response = client.post('/login', data={
                'username': 'testuser',
                'password': 'testpassword'
            }, follow_redirects=True)
            
            # Assert login was successful (optional but good for debugging setup)
            assert login_response.status_code == 200 # Should be 200 if redirected to dashboard
            assert b'Welcome, testuser!' in login_response.data or b'User Dashboard' in login_response.data

        yield client # This client will now have a logged-in session

        with app.app_context():
            db.session.remove()
            db.drop_all() # Clean up database after all tests in the module

def test_pnr_validation_required(test_client):
    """Test that PNR is required for complaint submission."""
    response = test_client.post('/complaint', data={
        'pnr_no': '', # Missing PNR
        'age': '30',
        'additional_info': 'Test complaint without PNR'
    }, follow_redirects=True)
    
    # Assert that it redirects back to the complaint page
    assert response.request.path == '/complaint'
    # Corrected assertion string based on your app.py:
    assert b'PNR Number is required.' in response.data
    assert b'complaint_error' in response.data # Check for the specific category

def test_pnr_validation_format(test_client):
    """Test that PNR must be a 10-digit number."""
    response = test_client.post('/complaint', data={
        'pnr_no': 'abc123456', # Invalid format (not all digits)
        'age': '30',
        'additional_info': 'Test complaint with invalid PNR format'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'PNR Number must be a 10-digit number.' in response.data
    assert b'complaint_error' in response.data

def test_pnr_validation_length(test_client):
    """Test that PNR must be a 10-digit number (length check)."""
    response = test_client.post('/complaint', data={
        'pnr_no': '12345', # Too short
        'age': '30',
        'additional_info': 'Test complaint with short PNR'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'PNR Number must be a 10-digit number.' in response.data
    assert b'complaint_error' in response.data

def test_age_validation_required(test_client):
    """Test that age is required for complaint submission."""
    response = test_client.post('/complaint', data={
        'pnr_no': '1234567890',
        'age': '', # Missing age
        'additional_info': 'Test complaint without age'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'Age is required.' in response.data
    assert b'complaint_error' in response.data

def test_age_validation_invalid_type(test_client):
    """Test that age must be a valid number."""
    response = test_client.post('/complaint', data={
        'pnr_no': '1234567890',
        'age': 'abc', # Invalid age type
        'additional_info': 'Test complaint with invalid age type'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'Invalid age. Please enter a valid number.' in response.data
    assert b'complaint_error' in response.data

def test_age_validation_out_of_range_low(test_client):
    """Test that age must be between 1 and 120 (too low)."""
    response = test_client.post('/complaint', data={
        'pnr_no': '1234567890',
        'age': '0', # Out of range (low)
        'additional_info': 'Test complaint with age too low'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'Age must be between 1 and 120.' in response.data
    assert b'complaint_error' in response.data

def test_age_validation_out_of_range_high(test_client):
    """Test that age must be between 1 and 120 (too high)."""
    response = test_client.post('/complaint', data={
        'pnr_no': '1234567890',
        'age': '121', # Out of range (high)
        'additional_info': 'Test complaint with age too high'
    }, follow_redirects=True)
    assert response.request.path == '/complaint'
    assert b'Age must be between 1 and 120.' in response.data
    assert b'complaint_error' in response.data


def test_valid_complaint_submission(test_client):
    """Test successful complaint submission."""
    # This test relies on you adding the flash message for success in app.py
    # as described above.
    response = test_client.post('/complaint', data={
        'pnr_no': '0987654321', # Use a unique PNR for each successful test if possible or clear DB between tests
        'age': '25',
        'additional_info': 'This is a valid complaint for E2E testing.'
    }, follow_redirects=True)
    
    # Assert that it redirects to user_dashboard after success
    assert response.request.path == '/user_dashboard'
    # Assert the specific success message content and category
    assert b'Complaint submitted successfully and assigned to' in response.data
    assert b'complaint_success' in response.data

    # Verify that the complaint was added to the database
    with app.app_context():
        complaint = Complaint.query.filter_by(pnr_no='0987654321').first()
        assert complaint is not None
        assert complaint.additional_info == 'This is a valid complaint for E2E testing.'
        assert complaint.status == 'Unsolved' # Assuming default status
        assert complaint.user_id == current_user.id # Ensure it's linked to the logged-in user


def test_complaint_get_request(test_client):
    """Test that the complaint page loads on GET request (requires login)."""
    response = test_client.get('/complaint')
    assert response.status_code == 200
    assert b'<title>File a Complaint</title>' in response.data