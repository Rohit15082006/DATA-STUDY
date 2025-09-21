# app.py
from flask import Flask, render_template, request, jsonify, send_file
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
import wavio
import re
import os
import json
import hashlib
import time
import threading
from werkzeug.utils import secure_filename
import uuid

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
print(f"Template folder path: {app.template_folder}")
print(f"Absolute template path: {os.path.abspath(app.template_folder)}")
print(f"Files in template folder: {os.listdir(app.template_folder)}")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class InterviewCoach:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(stopwords.words('english'))

        self.professional_words = {
            'accomplished', 'achieved', 'analyzed', 'coordinated', 'created',
            'delivered', 'developed', 'enhanced', 'executed', 'improved',
            'initiated', 'launched', 'managed', 'optimized', 'organized',
            'planned', 'resolved', 'spearheaded', 'streamlined', 'success'
        }

        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'so', 'well', 'just', 'stuff', 'things'
        }

    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Transcription complete!")
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
                return ""

    def analyze_tone(self, text):
        if not text:
            return {
                'score': 0,
                'sentiment': 'neutral',
                'feedback': 'No speech detected to analyze tone.'
            }

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        feedback = ""
        if sentiment == 'positive':
            feedback = "Your tone is positive and enthusiastic, which is great for an interview. Keep up the energy!"
            if compound_score > 0.5:
                feedback += " However, be careful not to come across as overly enthusiastic as it might seem insincere."
        elif sentiment == 'negative':
            feedback = "Your tone comes across as somewhat negative. Try to use more positive language and emphasize your strengths and achievements."
        else:
            feedback = "Your tone is neutral. While this is professional, try to inject some enthusiasm when discussing your achievements or interest in the role."

        return {
            'score': compound_score,
            'sentiment': sentiment,
            'feedback': feedback
        }

    def analyze_word_choice(self, text):
        if not text:
            return {
                'professional_word_count': 0,
                'filler_word_count': 0,
                'professional_words_used': [],
                'filler_words_used': [],
                'feedback': 'No speech detected to analyze word choice.'
            }

        words = nltk.word_tokenize(text.lower())
        professional_words_used = [word for word in words if word in self.professional_words]
        filler_words_used = [filler for filler in self.filler_words if filler in text.lower()]

        feedback = ""
        if professional_words_used:
            feedback += f"Good use of professional language! Words like {', '.join(professional_words_used[:3])} strengthen your responses. "
        else:
            feedback += "Consider incorporating more professional language to highlight your skills and achievements. "

        if filler_words_used:
            feedback += f"Try to reduce filler words/phrases like {', '.join(filler_words_used[:3])}. These can make you sound less confident."
        else:
            feedback += "You've done well avoiding filler words, which makes your speech sound more confident and prepared."

        return {
            'professional_word_count': len(professional_words_used),
            'filler_word_count': len(filler_words_used),
            'professional_words_used': professional_words_used,
            'filler_words_used': filler_words_used,
            'feedback': feedback
        }

    def analyze_confidence(self, text, tone_analysis):
        if not text:
            return {
                'confidence_score': 0,
                'feedback': 'No speech detected to analyze confidence.'
            }

        confidence_score = 5  # Base score out of 10
        sentiment_score = tone_analysis['score']
        if sentiment_score > 0:
            confidence_score += sentiment_score * 2
        elif sentiment_score < -0.2:
            confidence_score -= abs(sentiment_score) * 2

        hesitation_patterns = [
            r'\bI think\b', r'\bmaybe\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bI guess\b', r'\bsort of\b', r'\bkind of\b', r'\bI hope\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b'
        ]

        hesitation_count = sum(len(re.findall(pattern, text.lower())) for pattern in hesitation_patterns)
        confidence_score -= hesitation_count * 0.5

        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0

        if avg_sentence_length > 20:
            confidence_score += 1
        elif avg_sentence_length < 8:
            confidence_score -= 1

        confidence_score = max(0, min(10, confidence_score))

        if confidence_score >= 8:
            feedback = "You sound very confident. Your delivery is strong and assertive."
        elif confidence_score >= 6:
            feedback = "You sound reasonably confident. With a few adjustments, you could project even more authority."
        elif confidence_score >= 4:
            feedback = "Your confidence level seems moderate. Try speaking more assertively and avoiding hesitant language."
        else:
            feedback = "You may want to work on projecting more confidence. Try reducing hesitant phrases and speaking with more conviction."

        return {
            'confidence_score': confidence_score,
            'feedback': feedback
        }

    def provide_comprehensive_feedback(self, analysis_results):
        tone = analysis_results['tone']
        word_choice = analysis_results['word_choice']
        confidence = analysis_results['confidence']

        feedback_text = "\n" + "=" * 50 + "\n"
        feedback_text += "INTERVIEW RESPONSE EVALUATION\n"
        feedback_text += "=" * 50 + "\n\n"

        feedback_text += "TONE ANALYSIS:\n"
        feedback_text += f"Sentiment: {tone['sentiment']} (Score: {tone['score']:.2f})\n"
        feedback_text += f"Feedback: {tone['feedback']}\n\n"

        feedback_text += "WORD CHOICE ANALYSIS:\n"
        feedback_text += f"Professional words used: {word_choice['professional_word_count']}\n"
        if word_choice['professional_words_used']:
            feedback_text += f"Examples: {', '.join(list(set(word_choice['professional_words_used']))[:5])}\n"

        feedback_text += f"Filler words/phrases used: {word_choice['filler_word_count']}\n"
        if word_choice['filler_words_used']:
            feedback_text += f"Examples: {', '.join(list(set(word_choice['filler_words_used']))[:5])}\n"

        feedback_text += f"Feedback: {word_choice['feedback']}\n\n"

        feedback_text += "CONFIDENCE ASSESSMENT:\n"
        feedback_text += f"Confidence Score: {confidence['confidence_score']:.1f}/10\n"
        feedback_text += f"Feedback: {confidence['feedback']}\n\n"

        avg_score = (tone['score'] + 1) * 5 + confidence['confidence_score']
        avg_score /= 2

        if avg_score >= 8:
            feedback_text += "Excellent interview response! You presented yourself very well.\n"
        elif avg_score >= 6:
            feedback_text += "Good interview response. With some minor improvements, you'll make an even stronger impression.\n"
        elif avg_score >= 4:
            feedback_text += "Acceptable interview response. Focus on the improvement areas mentioned above.\n"
        else:
            feedback_text += "Your interview response needs improvement. Consider practicing more with the suggestions provided.\n"

        feedback_text += "\nAREAS TO FOCUS ON:\n"
        improvement_areas = []

        if tone['score'] < 0:
            improvement_areas.append("Using more positive language")
        if word_choice['filler_word_count'] > 3:
            improvement_areas.append("Reducing filler words/phrases")
        if word_choice['professional_word_count'] < 2:
            improvement_areas.append("Incorporating more professional vocabulary")
        if confidence['confidence_score'] < 5:
            improvement_areas.append("Building confidence in delivery")

        if improvement_areas:
            for i, area in enumerate(improvement_areas, 1):
                feedback_text += f"{i}. {area}\n"
        else:
            feedback_text += "Great job! Keep practicing to maintain your strong performance.\n"

        feedback_text += "=" * 50 + "\n"

        return feedback_text

    def analyze_text_input(self, text):
        tone_analysis = self.analyze_tone(text)
        word_choice_analysis = self.analyze_word_choice(text)
        confidence_analysis = self.analyze_confidence(text, tone_analysis)

        analysis_results = {
            'tone': tone_analysis,
            'word_choice': word_choice_analysis,
            'confidence': confidence_analysis,
            'text': text
        }

        return analysis_results

# Initialize the coach
coach = InterviewCoach()

# User management
def load_users():
    """Load users from file or create empty dict if file doesn't exist"""
    users_file = os.path.join(app.config['UPLOAD_FOLDER'], 'users.json')
    if os.path.exists(users_file):
        try:
            with open(users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to file"""
    users_file = os.path.join(app.config['UPLOAD_FOLDER'], 'users.json')
    with open(users_file, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    """Simple password hashing using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_username(username):
    """Validates that username is at least 8 characters and alphanumeric"""
    if not username:
        return False, "Username cannot be empty"
    
    if len(username) < 8:
        return False, "Username must be at least 8 characters long"
    
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return False, "Username can only contain letters and numbers"
    
    return True, "Valid username"

def validate_password(password):
    """Validates that password meets complex requirements"""
    if not password:
        return False, "Password cannot be empty"
    
    # Check length
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    # Check for uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    # Check for lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    # Check for digit
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    
    # Check for special character
    if not re.search(r'[!@#$%^&*\-_+=?]', password):
        return False, "Password must contain at least one special character (!@#$%^&*-_+=?)"
    
    return True, "Valid password"

def validate_email(email):
    """Validates email format"""
    if not email:
        return False, "Email cannot be empty"
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False, "Invalid email format"
    
    return True, "Valid email"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        users = load_users()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Please enter both username and password'})
        
        # Check if username exists and password matches
        if username in users and users[username]["password"] == hash_password(password):
            return jsonify({'success': True, 'message': f'Welcome back, {username}!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        
        users = load_users()
        
        # Validate fullname
        if not fullname:
            return jsonify({'success': False, 'message': 'Please enter your full name'})
        
        # Validate username
        valid_username, username_msg = validate_username(username)
        if not valid_username:
            return jsonify({'success': False, 'message': username_msg})
        
        # Check if username already exists
        if username in users:
            return jsonify({'success': False, 'message': 'Username already exists. Please choose another.'})
        
        # Validate email
        valid_email, email_msg = validate_email(email)
        if not valid_email:
            return jsonify({'success': False, 'message': email_msg})
        
        # Validate password
        valid_password, password_msg = validate_password(password)
        if not valid_password:
            return jsonify({'success': False, 'message': password_msg})
        
        # Validate password confirmation
        if password != confirm:
            return jsonify({'success': False, 'message': 'Passwords do not match'})
        
        # Create new user
        users[username] = {
            "fullname": fullname,
            "email": email,
            "password": hash_password(password)
        }
        
        # Save to file
        save_users(users)
        
        return jsonify({'success': True, 'message': 'Account created successfully. You can now login.'})
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    question = request.form.get('question', '')
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    filename = secure_filename(f"{uuid.uuid4()}.wav")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    # Analyze the audio
    text = coach.transcribe_audio(filepath)
    
    if not text:
        os.remove(filepath)
        return jsonify({'error': 'Could not transcribe audio. Please try again.'}), 400
    
    # Perform analysis
    tone_analysis = coach.analyze_tone(text)
    word_choice_analysis = coach.analyze_word_choice(text)
    confidence_analysis = coach.analyze_confidence(text, tone_analysis)
    
    analysis_results = {
        'tone': tone_analysis,
        'word_choice': word_choice_analysis,
        'confidence': confidence_analysis,
        'text': text,
        'question': question
    }
    
    # Generate comprehensive feedback
    feedback = coach.provide_comprehensive_feedback(analysis_results)
    
    # Save the analysis results
    analysis_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.json")
    with open(analysis_filename, 'w') as f:
        json.dump({
            'question': question,
            'transcription': text,
            'analysis': analysis_results,
            'feedback': feedback
        }, f)
    
    # Clean up the audio file
    os.remove(filepath)
    
    return jsonify({
        'success': True,
        'question': question,
        'transcription': text,
        'analysis': analysis_results,
        'feedback': feedback
    })

@app.route('/get_questions')
def get_questions():
    questions = [
        "Tell me about yourself.",
        "What are your strengths?",
        "What are your weaknesses?",
        "Why do you want to work for this company?",
        "Where do you see yourself in five years?",
        "Why should we hire you?",
        "Tell me about a challenge you faced and how you overcame it.",
        "What's your greatest professional achievement?",
        "How do you handle stress and pressure?",
        "Do you have any questions for me?"
    ]
    return jsonify({'questions': questions})

if __name__ == '__main__':
    app.run(debug=True)