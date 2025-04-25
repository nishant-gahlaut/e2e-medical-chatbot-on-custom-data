from flask import jsonify, url_for
from src.helper import format_username

def handle_set_name(request, session):
    """Handles the logic for setting the user's name."""
    data = request.get_json()
    name = data.get('name')
    if not name:
        return False, {"success": False, "error": "Name is required"}, 400
    
    user_id = format_username(name)
    if not user_id:
        return False, {"success": False, "error": "Invalid name format"}, 400
    
    session['user_id'] = user_id
    
    return True, {"success": True, "redirect": url_for('chat')}, 200

def handle_get_user_name(session):
    """Handles the logic for getting the user's name."""
    return jsonify({"name": session.get('user_id', '')}) 