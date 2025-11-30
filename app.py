from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from bot_engine import StudentBot

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize bot
print("Initializing bot...")
bot = StudentBot()
print("Bot ready!")

# Simple in-memory store for student info per session
student_sessions = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.json
        session_id = data.get('session_id')  # unique per user/browser
        message = data.get('message', '').strip()
        
        # Retrieve student info for this session
        student_info = student_sessions.get(session_id, {})
        student_name = student_info.get('student_name')
        student_phone = student_info.get('student_phone')
        
        # If student info missing, expect it in this message
        if not student_name or not student_phone:
            # Expect frontend to send these
            student_name = data.get('student_name')
            student_phone = data.get('student_phone')
            
            if not student_name or not student_phone:
                return jsonify({
                    'response': "Hi! Before we start, please provide your name and phone number.",
                    'status': 'need_info'
                })
            
            # Save info for session
            student_sessions[session_id] = {
                'student_name': student_name,
                'student_phone': student_phone
            }
            return jsonify({
                'response': f"✅ Chat started with {student_name}. You can now ask your questions.",
                'status': 'success'
            })
        
        # Normal chat flow
        response = bot.handle_message(student_phone, student_name, message)
        
        # Check handoff
        needs_handoff = bot.check_handoff_needed(message, response)
        
        return jsonify({
            'response': response,
            'needs_handoff': needs_handoff,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e),
            'response': "Sorry, I'm having trouble right now. Try again?",
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
