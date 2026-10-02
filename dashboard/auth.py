import hashlib
import streamlit as st

def hash_password(password: str) -> str:
    """Generate SHA-256 hash for checking against secrets."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def check_credentials(username, password):
    """Check credentials from .streamlit/secrets.toml"""
    if "users" not in st.secrets:
        st.error("Authentication configuration missing.")
        return False, None
    
    users = st.secrets["users"]
    if username in users:
        user_info = users[username]
        if hash_password(password) == user_info["password_hash"]:
            return True, user_info["role"]
    
    return False, None

def login():
    """Renders login form and populates session_state."""
    st.title("LandSentry Login")
    with st.form("login_form"):
        username = st.text_input("Username").strip().lower()
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, role = check_credentials(username, password)
            if success:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                st.success(f"Welcome {username}! Loading...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

def logout():
    """Clears session and forces rerun."""
    for key in ["logged_in", "username", "role"]:
        st.session_state.pop(key, None)
    st.rerun()

def is_authorized(allowed_roles):
    """Verify current user role is authorized."""
    if not st.session_state.get("logged_in"):
        return False
    current_role = st.session_state.get("role")
    if current_role == "Admin":
        return True
    return current_role in allowed_roles
