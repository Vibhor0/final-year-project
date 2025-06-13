import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time # Used for small debugging sleeps, prefer explicit waits for robustness

# --- IMPORTANT: Configure your BASE_URL ---
BASE_URL = "https://final-year-project-dciy.onrender.com" # Removed trailing slash to prevent double slash issue

# --- Test User Credentials (ensure this user exists on your deployed app) ---
TEST_USERNAME = "user1"
TEST_PASSWORD = "1234" # Use the actual password for testuser

@pytest.fixture(scope="module")
def browser():
    """
    Sets up a Chrome browser instance for E2E tests.
    Ensure ChromeDriver is in your system's PATH.
    """
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Uncomment THIS LINE to run with a visible browser
    options.add_argument("--no-sandbox") # Required for some environments like Docker/CI
    options.add_argument("--disable-dev-shm-usage") # Overcome limited resource problems
    
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1200, 800) # Set a consistent window size
    driver.implicitly_wait(0) # IMPORTANT: Set implicit wait to 0 when using explicit waits
    yield driver
    driver.quit() # Close the browser after all tests in the module

def test_user_login_and_complaint_submission(browser):
    """
    Tests the full flow: login, navigate to complaint, submit, and verify success.
    """
    print(f"\n--- Starting test: {test_user_login_and_complaint_submission.__name__} ---")

    # 1. Navigate to the login page
    browser.get(f"{BASE_URL}/login")
    print(f"1. Navigated to login page: {browser.current_url}")
    print(f"   Page title: {browser.title}") # Added to check if title loads

    # Small sleep to ensure initial page content starts loading (helpful for slow connections/cold starts)
    time.sleep(3) 

    # Wait for the username input field to be present and visible
    try:
        username_input = WebDriverWait(browser, 25).until( # Increased to 25 seconds
            EC.visibility_of_element_located((By.ID, "username"))
        )
        print("2. Username input found.")
    except Exception as e:
        print(f"Timeout: Username input not found. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        raise e

    # 2. Perform Login
    username_input.send_keys(TEST_USERNAME)
    browser.find_element(By.ID, "password").send_keys(TEST_PASSWORD)
    print("3. Entered credentials.")
    
    submit_button = browser.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_button.click()
    print("4. Clicked submit button.")

    # 3. Verify successful login and redirect to user_dashboard
    print(f"   Current URL before redirect wait: {browser.current_url}") # Check URL just before waiting
    try:
        WebDriverWait(browser, 30).until( # Increased to 30 seconds for post-login redirect
            EC.url_contains(f"{BASE_URL}/user_dashboard") # Changed to url_contains for more flexibility
        )
        WebDriverWait(browser, 5).until( # Added to confirm page content loaded
            EC.title_contains("User Dashboard")
        )
        print(f"5. Successfully redirected to user dashboard: {browser.current_url}")
    except Exception as e:
        print(f"Timeout: Did not redirect to user_dashboard. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        if "Bad credentials" in browser.page_source or "Invalid username or password" in browser.page_source:
             print("Login likely failed: Check your test_user credentials on Render.")
        raise e
    
    assert "User Dashboard" in browser.title # Check page title explicitly
    print("6. User Dashboard title confirmed.")

    # 4. Navigate to the complaint page
    browser.get(f"{BASE_URL}/complaint")
    print(f"7. Navigated to complaint page: {browser.current_url}")
    print(f"   Page title: {browser.title}") # Added to check if title loads

    # Wait for complaint page elements to load
    try:
        pnr_input = WebDriverWait(browser, 20).until( # Increased to 20 seconds
            EC.visibility_of_element_located((By.ID, "pnr_no"))
        )
        print("8. PNR input found on complaint page.")
    except Exception as e:
        print(f"Timeout: PNR input not found on complaint page. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        raise e

    # 5. Fill the complaint form
    pnr_input.send_keys("1234567890") # Use a unique PNR if possible for clean data
    browser.find_element(By.ID, "age").send_keys("30")
    browser.find_element(By.ID, "additional_info").send_keys("This is an E2E test complaint from Selenium.")
    print("9. Filled complaint form.")

    # 6. Submit the form
    submit_complaint_button = browser.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_complaint_button.click()
    print("10. Clicked submit complaint button.")

    # 7. Assert successful submission and redirect
    print(f"   Current URL before redirect wait: {browser.current_url}") # Check URL just before waiting
    try:
        WebDriverWait(browser, 30).until( # Increased to 30 seconds for post-complaint submission redirect
            EC.url_contains(f"{BASE_URL}/user_dashboard") # Changed to url_contains
        )
        WebDriverWait(browser, 5).until( # Added to confirm page content loaded
            EC.title_contains("User Dashboard")
        )
        print(f"11. Successfully redirected to user dashboard after complaint: {browser.current_url}")
    except Exception as e:
        print(f"Timeout: Did not redirect to user_dashboard after complaint. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        if "PNR Number is required." in browser.page_source: # Example check for error on same page
            print("Complaint submission likely failed with a validation error, remaining on complaint page.")
        raise e

    # Assert that the success message is displayed on the user dashboard
    success_message_locator = (By.CLASS_NAME, "alert-complaint_success") # Check for the Bootstrap alert class
    try:
        WebDriverWait(browser, 15).until( # Increased to 15 seconds
            EC.visibility_of_element_located(success_message_locator)
        )
        success_message_text = browser.find_element(*success_message_locator).text
        assert "Complaint submitted successfully and assigned to" in success_message_text
        print(f"12. Success message found: '{success_message_text}'")
    except Exception as e:
        print(f"Timeout: Success message not found. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        print("Verify your app.py includes 'flash(f'Complaint submitted successfully...' and your dashboard.html renders flashed messages with class 'alert-complaint_success'.")
        raise e
    
    print(f"--- Test {test_user_login_and_complaint_submission.__name__} PASSED ---")


def test_pnr_validation_error_on_complaint_page(browser):
    """
    Tests that a PNR validation error is displayed correctly.
    """
    print(f"\n--- Starting test: {test_pnr_validation_error_on_complaint_page.__name__} ---")

    # Ensure logged in for this test as well (re-login or use a different fixture scope)
    browser.get(f"{BASE_URL}/login")
    print(f"1. Navigated to login page for PNR test: {browser.current_url}")

    try:
        WebDriverWait(browser, 25).until(EC.visibility_of_element_located((By.ID, "username"))).send_keys(TEST_USERNAME)
        browser.find_element(By.ID, "password").send_keys(TEST_PASSWORD)
        browser.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        print("2. Logged in for PNR test.")
    except Exception as e:
        print(f"Timeout: Login failed for PNR test. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        raise e

    print(f"   Current URL before redirect wait: {browser.current_url}") # Check URL just before waiting
    try:
        WebDriverWait(browser, 30).until( # Increased to 30 seconds
            EC.url_contains(f"{BASE_URL}/user_dashboard") # Changed to url_contains
        )
        WebDriverWait(browser, 5).until( # Added to confirm page content loaded
            EC.title_contains("User Dashboard")
        )
        print("3. Redirected to user dashboard for PNR test.")
    except Exception as e:
        print(f"Timeout: Did not redirect to user_dashboard after login for PNR test. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        raise e

    # Navigate to complaint page
    browser.get(f"{BASE_URL}/complaint")
    print(f"4. Navigated to complaint page for PNR test: {browser.current_url}")
    print(f"   Page title: {browser.title}") # Added to check if title loads

    try:
        WebDriverWait(browser, 20).until(EC.visibility_of_element_located((By.ID, "pnr_no")))
        print("5. PNR input found for PNR test.")
    except Exception as e:
        print(f"Timeout: PNR input not found on complaint page for PNR test. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        raise e

    # Submit with missing PNR
    browser.find_element(By.ID, "pnr_no").send_keys("") # Empty PNR
    browser.find_element(By.ID, "age").send_keys("25")
    browser.find_element(By.ID, "additional_info").send_keys("Test with missing PNR.")
    browser.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    print("6. Submitted form with missing PNR.")

    # Expect to stay on the complaint page and see the error message
    print(f"   Current URL before redirect wait: {browser.current_url}") # Check URL just before waiting
    try:
        WebDriverWait(browser, 20).until( # Increased to 20 seconds
            EC.url_contains(f"{BASE_URL}/complaint") # Changed to url_contains
        )
        WebDriverWait(browser, 5).until( # Added to confirm page content loaded
            EC.title_contains("File a Complaint")
        )
        print(f"7. Remained on complaint page after invalid submission: {browser.current_url}")
    except Exception as e:
        print(f"Timeout: Did not remain on complaint page for PNR test. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        print("Verify your Flask app logic for PNR validation and redirection.")
        raise e

    error_message_locator = (By.CLASS_NAME, "alert-complaint_error")
    try:
        WebDriverWait(browser, 15).until( # Increased to 15 seconds
            EC.visibility_of_element_located(error_message_locator)
        )
        error_message_text = browser.find_element(*error_message_locator).text
        assert "PNR Number is required." in error_message_text
        print(f"8. Error message found: '{error_message_text}'")
    except Exception as e:
        print(f"Timeout: Error message not found. Current URL: {browser.current_url}, Page source (partial):\n{browser.page_source[:500]}...")
        print("Verify your app.py includes 'flash('PNR Number is required.', 'complaint_error')' and your complaint.html renders flashed messages with class 'alert-complaint_error'.")
        raise e

    print(f"--- Test {test_pnr_validation_error_on_complaint_page.__name__} PASSED ---")