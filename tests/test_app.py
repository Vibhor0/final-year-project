import pytest
from app import app, db, User, Complaint, Feedback # Make sure Feedback is imported
from werkzeug.security import generate_password_hash, check_password_hash # For creating/checking test user password
from flask_login import current_user # Still imported for context, but usage refined
from datetime import datetime
from unittest.mock import patch # Needed for mocking external dependencies

# Configure app for testing
@pytest.fixture(scope='function') # Scope remains 'function' for isolation
def test_client():
    # Set app config for testing
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:' # Use in-memory SQLite for tests
    app.config['SECRET_KEY'] = 'a_super_secret_test_key_for_session' # Crucial for session (flash messages)
    app.config['WTF_CSRF_ENABLED'] = False # Disable CSRF for easier testing

    with app.test_client() as client:
        with app.app_context():
            db.create_all() # Create tables for the in-memory db for each test

            # Create test users (these are added and committed in this session)
            hashed_password_user = generate_password_hash('testpassword', method='pbkdf2:sha256')
            test_user = User(username='testuser', password=hashed_password_user, role='user', department='General')
            db.session.add(test_user)
            
            hashed_password_employee = generate_password_hash('employeepass', method='pbkdf2:sha256')
            employee_user = User(username='employeeuser', password=hashed_password_employee, role='employee', department='EmployeeDept')
            db.session.add(employee_user)

            hashed_password_admin = generate_password_hash('adminpass', method='pbkdf2:sha256')
            admin_user = User(username='adminuser', password=hashed_password_admin, role='admin', department='AdminDept')
            db.session.add(admin_user)

            db.session.commit()
            
            # DO NOT attach User objects to client here.
            # They will be re-queried in each test function as needed.
            
        yield client # Yield the client after setup

        # Teardown: Clean up the database for 'function' scope
        with app.app_context():
            db.session.remove() # Clean up session after each test
            db.drop_all() # Drop all tables after each test


# Helper function to log in a user for tests
def login_test_user(client, username, password):
    # This helper function takes string username/password, so no detached object issue here
    return client.post('/login', data={
        'username': username,
        'password': password
    }, follow_redirects=True)

# Helper function to logout a user
def logout_test_user(client):
    return client.get('/logout', follow_redirects=True)

# Test functions
def test_login_page_get_request(test_client):
    """Test that the login page loads on GET request."""
    response = test_client.get('/login')
    assert response.status_code == 200
    assert b'Login' in response.data
    assert b'username' in response.data
    assert b'password' in response.data

def test_login_page_post_request_success(test_client):
    """Test successful login."""
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        response = login_test_user(test_client, user.username, 'testpassword')
    
    assert response.status_code == 200 # Should redirect to user_dashboard, which returns 200
    assert response.request.path == '/user_dashboard'
    assert b'User Dashboard' in response.data # Check for content on dashboard

def test_login_page_post_request_invalid_credentials(test_client):
    """Test failed login with invalid credentials."""
    response = login_test_user(test_client, 'nonexistentuser', 'wrongpassword')
    assert response.status_code == 200 # Still on login page, but should show flash message
    assert response.request.path == '/login'
    assert b'Invalid username or password' in response.data # Assuming this flash message

def test_logout(test_client):
    """Test user logout functionality."""
    # First, log in a user using the fresh object
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')
    
    assert test_client.get('/user_dashboard').request.path == '/user_dashboard' # Confirm logged in

    # Then, log out
    response = logout_test_user(test_client)
    assert response.status_code == 200
    assert response.request.path == '/user_dashboard' # After logout, it redirects to user_dashboard
    
    # More robust: try to access a login_required page and ensure it redirects to login
    response_protected = test_client.get('/complaint', follow_redirects=False)
    assert response_protected.status_code == 302 # Should redirect
    assert '/login' in response_protected.headers['Location'] # To login page

def test_complaint_get_request(test_client):
    """Test that the complaint page loads on GET request (requires login)."""
    # Ensure user is logged in
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')
    
    response = test_client.get('/complaint')
    assert response.status_code == 200
    assert b'File a Complaint' in response.data
    assert b'pnr_no' in response.data

def test_complaint_submission_success(test_client):
    """Test successful complaint submission."""
    with app.app_context(): # Ensure app context for querying user and later complaint
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')
    
    with patch('app.department_classifier_model') as mock_classifier:
        with patch('app.tf_vectorizer') as mock_vectorizer:
            mock_classifier.predict.return_value = ['General'] # Mock the classification result
            mock_vectorizer.transform.return_value = 'mocked_vector' # Mock vectorizer output

            response = test_client.post('/complaint', data={
                'pnr_no': 'COMP1234567890', # Use a unique PNR for each test if possible
                'age': '30',
                'additional_info': 'This is a successful complaint from test_app.py.'
            }, follow_redirects=True)
            
            assert response.status_code == 200
            assert response.request.path == '/user_dashboard'
            assert b'Complaint submitted successfully' in response.data
            assert b'testuser' in response.data # Verify user specific content on dashboard

            with app.app_context(): # Re-enter context for DB operations if needed, or ensure context is active
                complaint = Complaint.query.filter_by(pnr_no='COMP1234567890').first()
                assert complaint is not None
                assert complaint.user_id == user.id # <<-- IMPORTANT CHANGE: Use ID from re-queried user
                assert complaint.status == 'Unsolved' # Assuming default status

def test_complaint_validation_error(test_client):
    """Test complaint submission with a validation error (missing PNR)."""
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')

    response = test_client.post('/complaint', data={
        'pnr_no': '', # Empty PNR
        'age': '25',
        'additional_info': 'This complaint should trigger a validation error.'
    }, follow_redirects=True) # follow_redirects=True to handle cases where it might redirect to login if session expires

    # Assert that it stays on the complaint page and shows the error message
    assert response.status_code == 200
    assert response.request.path == '/complaint' # Should remain on the complaint page
    assert b'PNR Number is required.' in response.data # Assuming this flash message content
    assert b'alert-complaint_error' in response.data # Assuming this class for error messages

# --- NEW TEST FUNCTIONS START HERE ---

def test_user_dashboard_view_complaints(test_client):
    """
    Test that a logged-in user can view their submitted complaints on the user dashboard.
    """
    # 1. Login as testuser
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')

    # 2. Submit a unique complaint
    pnr_for_dashboard = 'DASHBOARD123'
    additional_info_dashboard = 'Complaint to verify on dashboard.'
    
    with patch('app.department_classifier_model') as mock_classifier:
        with patch('app.tf_vectorizer') as mock_vectorizer:
            mock_classifier.predict.return_value = ['General']
            mock_vectorizer.transform.return_value = 'mocked_vector'
            test_client.post('/complaint', data={
                'pnr_no': pnr_for_dashboard,
                'age': '40',
                'additional_info': additional_info_dashboard
            }, follow_redirects=True) # Redirects to user_dashboard

    # 3. Access user dashboard
    response = test_client.get('/user_dashboard')
    assert response.status_code == 200
    assert b'User Dashboard' in response.data
    
    # 4. Assert the complaint is visible on the dashboard
    assert pnr_for_dashboard.encode('utf-8') in response.data
    assert additional_info_dashboard.encode('utf-8') in response.data
    assert b'Unsolved' in response.data # Check initial status

def test_employee_login_and_dashboard_access(test_client):
    """
    Test that an employee can log in and access their dashboard.
    """
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        employee = User.query.filter_by(username='employeeuser').first()
        response = login_test_user(test_client, employee.username, 'employeepass')
    
    assert response.status_code == 200
    assert response.request.path == '/employee_dashboard'
    assert b'Employee Dashboard' in response.data
    assert b'Assigned Complaints' in response.data

def test_mark_complaint_as_solved(test_client):
    """
    Test that an employee can mark an assigned complaint as solved.
    """
    # 1. Login as a regular user and submit a complaint
    with app.app_context(): # Ensure app context for creating and assigning users/complaints
        user = User.query.filter_by(username='testuser').first()
        employee = User.query.filter_by(username='employeeuser').first()
        login_test_user(test_client, user.username, 'testpassword')
    
    pnr_to_solve = 'SOLVE987654321'
    additional_info_to_solve = 'Complaint to be marked as solved.'
    
    complaint_id_to_solve = None # Initialize outside context

    with app.app_context(): # Need app context to assign employee to complaint
        # Use patch to mock department classifier model if it's needed during complaint submission
        with patch('app.department_classifier_model') as mock_classifier:
            with patch('app.tf_vectorizer') as mock_vectorizer:
                # Mock return values for department classification
                mock_classifier.predict.return_value = ['General'] # Assign to General department
                mock_vectorizer.transform.return_value = 'mocked_vector' # Mock vectorizer output
                
                # Submit complaint (this will create it in the DB)
                test_client.post('/complaint', data={
                    'pnr_no': pnr_to_solve,
                    'age': '35',
                    'additional_info': additional_info_to_solve
                }, follow_redirects=True)

                # Retrieve the newly created complaint within the same context
                complaint = Complaint.query.filter_by(pnr_no=pnr_to_solve).first()
                assert complaint is not None
                
                # Manually assign to the employee for testing if app doesn't do it automatically based on department
                complaint.assigned_employee_id = employee.id # <<-- IMPORTANT CHANGE: Use ID from re-queried employee
                db.session.add(complaint)
                db.session.commit()
                complaint_id_to_solve = complaint.unique_id # Get the unique ID for the URL
    
    logout_test_user(test_client) # Logout test user

    # 2. Login as employee
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query employee within app context
        employee = User.query.filter_by(username='employeeuser').first()
        login_test_user(test_client, employee.username, 'employeepass')
    
    # 3. Mark the complaint as solved
    # Use the unique_id obtained from the database for the URL
    response_solve = test_client.post(f'/mark_as_solved/{complaint_id_to_solve}', follow_redirects=True)
    assert response_solve.status_code == 200
    assert response_solve.request.path == '/employee_dashboard' # Should redirect back to employee dashboard

    # 4. Verify the complaint status in the database
    with app.app_context():
        updated_complaint = Complaint.query.filter_by(unique_id=complaint_id_to_solve).first()
        assert updated_complaint.status == 'Solved'
        assert b'Complaint marked as solved' in response_solve.data # Check for success message

def test_admin_login_and_database_access(test_client):
    """
    Test that an admin can log in and access the database view.
    """
    # 1. Login as admin
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query admin within app context
        admin = User.query.filter_by(username='adminuser').first()
        response = login_test_user(test_client, admin.username, 'adminpass')
    
    assert response.status_code == 200
    assert response.request.path == '/admin_dashboard' # Assuming admin logs into admin_dashboard

    # 2. Access the view_database page
    response_db = test_client.get('/view_database')
    assert response_db.status_code == 200
    assert b'Database View' in response_db.data
    assert b'User Database' in response_db.data
    assert b'Complaints Database' in response_db.data
    assert b'Feedbacks Database' in response_db.data
    
    # Assert that the created users are visible in the database view by querying them again
    with app.app_context():
        test_user_db = User.query.filter_by(username='testuser').first()
        employee_user_db = User.query.filter_by(username='employeeuser').first()
        admin_user_db = User.query.filter_by(username='adminuser').first()
    
    assert test_user_db.username.encode('utf-8') in response_db.data
    assert employee_user_db.username.encode('utf-8') in response_db.data
    assert admin_user_db.username.encode('utf-8') in response_db.data


def test_feedback_submission(test_client):
    """
    Test that a user can submit feedback for a solved complaint.
    """
    # 1. Create and solve a complaint (requires multiple roles)
    pnr_for_feedback = 'FEEDBACK999'
    complaint_unique_id = None
    complaint_db_id = None
    
    # Context to create and assign/solve complaint
    with app.app_context():
        # User logs in and creates complaint
        user = User.query.filter_by(username='testuser').first()
        employee = User.query.filter_by(username='employeeuser').first()
        
        login_test_user(test_client, user.username, 'testpassword')
        with patch('app.department_classifier_model') as mock_classifier:
            with patch('app.tf_vectorizer') as mock_vectorizer:
                mock_classifier.predict.return_value = ['General']
                mock_vectorizer.transform.return_value = 'mocked_vector'
                test_client.post('/complaint', data={
                    'pnr_no': pnr_for_feedback,
                    'age': '28',
                    'additional_info': 'Complaint for feedback test.'
                }, follow_redirects=True)
        
        # Get the created complaint and assign it to employee for solving
        complaint = Complaint.query.filter_by(pnr_no=pnr_for_feedback).first()
        assert complaint is not None
        complaint.assigned_employee_id = employee.id # <<-- IMPORTANT CHANGE: Use ID from re-queried employee
        db.session.add(complaint)
        db.session.commit()
        complaint_unique_id = complaint.unique_id # Get unique ID for URL

        # Employee logs in and marks it solved
        logout_test_user(test_client)
        login_test_user(test_client, employee.username, 'employeepass')
        test_client.post(f'/mark_as_solved/{complaint_unique_id}', follow_redirects=True)
        logout_test_user(test_client)

        # Confirm complaint is solved (re-query within context)
        solved_complaint = Complaint.query.filter_by(unique_id=complaint_unique_id).first()
        assert solved_complaint.status == 'Solved'
        # Get the database ID of the solved complaint for the feedback form URL
        complaint_db_id = solved_complaint.id

    # 2. User logs back in to submit feedback
    with app.app_context(): # <<-- IMPORTANT CHANGE: Query user within app context
        user = User.query.filter_by(username='testuser').first()
        login_test_user(test_client, user.username, 'testpassword')
    
    # Access the feedback form (assuming the URL is /feedback/<complaint_db_id>)
    feedback_form_url = f'/feedback/{complaint_db_id}' # Use database ID here
    response_get_feedback_form = test_client.get(feedback_form_url)
    assert response_get_feedback_form.status_code == 200
    assert b'Provide Feedback for Complaint' in response_get_feedback_form.data

    # 3. Submit feedback
    feedback_text_val = 'This is a test feedback message.'
    rating_val = '5'
    with patch('app.TextBlob') as mock_textblob:
        mock_textblob.return_value.sentiment.polarity = 0.8 # Mock positive sentiment
        mock_textblob.return_value.sentiment.subjectivity = 0.5
        
        response_submit_feedback = test_client.post(feedback_form_url, data={
            'feedback_text': feedback_text_val,
            'rating': rating_val
        }, follow_redirects=True)
    
    assert response_submit_feedback.status_code == 200
    assert response_submit_feedback.request.path == '/user_dashboard' # Assuming redirect to dashboard
    assert b'Feedback submitted successfully' in response_submit_feedback.data # Check flash message

    # 4. Verify feedback in the database
    with app.app_context():
        # Query using the complaint's database ID
        feedback = Feedback.query.filter_by(complaint_id=complaint_db_id).first()
        assert feedback is not None
        assert feedback.feedback_text == feedback_text_val
        assert feedback.rating == int(rating_val)
        assert feedback.sentiment == 'Positive' # Based on mock
