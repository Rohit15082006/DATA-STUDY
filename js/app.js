// Main Application Controller
class InterviewCoachApp {
    constructor() {
        // DOM Elements
        this.loginScreen = document.getElementById('login-screen');
        this.appScreen = document.getElementById('app-screen');
        this.loginContainer = document.querySelector('.login-container');
        this.registerContainer = document.querySelector('.register-container');
        
        // Login Elements
        this.loginBtn = document.getElementById('login-btn');
        this.showRegisterBtn = document.getElementById('show-register');
        this.showLoginBtn = document.getElementById('show-login');
        this.registerBtn = document.getElementById('register-btn');
        this.logoutBtn = document.getElementById('logout-btn');
        
        // User Input Fields
        this.usernameInput = document.getElementById('username');
        this.passwordInput = document.getElementById('password');
        this.regNameInput = document.getElementById('reg-name');
        this.regUsernameInput = document.getElementById('reg-username');
        this.regEmailInput = document.getElementById('reg-email');
        this.regPasswordInput = document.getElementById('reg-password');
        this.regConfirmInput = document.getElementById('reg-confirm');
        
        // App Elements
        this.welcomeUser = document.getElementById('welcome-user');
        this.questionDropdown = document.getElementById('question-dropdown');
        this.recordBtn = document.getElementById('record-btn');
        this.pauseBtn = document.getElementById('pause-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        this.timerDisplay = document.getElementById('timer-display');
        this.questionDisplay = document.getElementById('question-display');
        this.transcriptionDisplay = document.getElementById('transcription-display');
        this.analysisResults = document.getElementById('analysis-results');
        this.downloadBtn = document.getElementById('download-btn');
        
        // Modal Elements
        this.modal = document.getElementById('modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalMessage = document.getElementById('modal-message');
        this.modalConfirm = document.getElementById('modal-confirm');
        this.closeModal = document.querySelector('.close-modal');
        
        // Initialize modules
        this.recorder = new AudioRecorder(this);
        this.camera = new CameraHandler(this);
        this.analyzer = new ResponseAnalyzer(this);
        
        // Current state
        this.currentUser = null;
        this.currentAnalysis = null;
        this.isRecording = false;
        this.isPaused = false;
        this.remainingTime = 60;
        this.timerInterval = null;
        
        // Initialize event listeners
        this.initEventListeners();
    }

initEventListeners() {
        // Login/Register events
        this.loginBtn.addEventListener('click', () => this.handleLogin());
        this.showRegisterBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.showRegister();
        });
        this.showLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.showLogin();
        });
        this.registerBtn.addEventListener('click', () => this.handleRegister());
        this.logoutBtn.addEventListener('click', () => this.handleLogout());
        
        // Recording events
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.pauseBtn.addEventListener('click', () => this.togglePause());
        this.cancelBtn.addEventListener('click', () => this.cancelRecording());
        
        // Modal events
        this.modalConfirm.addEventListener('click', () => this.hideModal());
        this.closeModal.addEventListener('click', () => this.hideModal());
        
        // Download report
        this.downloadBtn.addEventListener('click', () => this.downloadReport());
        
        // Allow Enter key to submit forms
        this.passwordInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleLogin();
        });
        this.regConfirmInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleRegister();
        });
    }
    
    // User Authentication Methods
    async handleLogin() {
        const username = this.usernameInput.value.trim();
        const password = this.passwordInput.value.trim();
        
        if (!username || !password) {
            this.showModal('Error', 'Please enter both username and password');
            return;
        }
        
        // In a real app, this would be an API call to your backend
        const users = JSON.parse(localStorage.getItem('interviewCoachUsers')) || {};
        
        if (users[username] && users[username].password === this.hashPassword(password)) {
            this.currentUser = {
                username: username,
                fullname: users[username].fullname,
                email: users[username].email
            };
            
            // Save current user to session
            sessionStorage.setItem('currentUser', JSON.stringify(this.currentUser));
            
            // Show main app
            this.showApp();
        } else {
            this.showModal('Error', 'Invalid username or password');
        }
    }
    
    async handleRegister() {
        const fullname = this.regNameInput.value.trim();
        const username = this.regUsernameInput.value.trim();
        const email = this.regEmailInput.value.trim();
        const password = this.regPasswordInput.value.trim();
        const confirm = this.regConfirmInput.value.trim();
        
        // Validate inputs
        if (!fullname) {
            this.showModal('Error', 'Please enter your full name');
            return;
        }
        
        // Validate username
        if (username.length < 8 || !/^[a-zA-Z0-9]+$/.test(username)) {
            this.showModal('Error', 'Username must be at least 8 characters (letters and numbers only)');
            return;
        }
        
        // Validate email
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            this.showModal('Error', 'Please enter a valid email address');
            return;
        }
        
        // Validate password
        if (password.length < 8 || 
            !/[A-Z]/.test(password) || 
            !/[a-z]/.test(password) || 
            !/[0-9]/.test(password) || 
            !/[!@#$%^&*\-_+=?]/.test(password)) {
            this.showModal('Error', 'Password must be at least 8 characters with uppercase, lowercase, number, and special character');
            return;
        }
        
        if (password !== confirm) {
            this.showModal('Error', 'Passwords do not match');
            return;
        }
        
        // Check if username exists
        const users = JSON.parse(localStorage.getItem('interviewCoachUsers')) || {};
        
        if (users[username]) {
            this.showModal('Error', 'Username already exists. Please choose another.');
            return;
        }
        
        // Create new user
        users[username] = {
            fullname: fullname,
            email: email,
            password: this.hashPassword(password)
        };
        
        // Save to localStorage
        localStorage.setItem('interviewCoachUsers', JSON.stringify(users));
        
        // Show success message
        this.showModal('Success', 'Account created successfully. You can now login.', () => {
            // Switch to login and pre-fill username
            this.showLogin();
            this.usernameInput.value = username;
            this.passwordInput.value = '';
            this.passwordInput.focus();
            
            // Clear registration form
            this.regNameInput.value = '';
            this.regUsernameInput.value = '';
            this.regEmailInput.value = '';
            this.regPasswordInput.value = '';
            this.regConfirmInput.value = '';
        });
    }
    
    handleLogout() {
        // Clear session
        sessionStorage.removeItem('currentUser');
        this.currentUser = null;
        
        // Stop any ongoing recording
        if (this.isRecording) {
            this.recorder.stopRecording();
        }
        
        // Stop camera
        this.camera.stopCamera();
        
        // Reset UI
        this.showLoginScreen();
    }
    
    hashPassword(password) {
        // Simple hash function for demo purposes
        // In a real app, use proper password hashing like bcrypt
        let hash = 0;
        for (let i = 0; i < password.length; i++) {
            const char = password.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash |= 0; // Convert to 32bit integer
        }
        return hash.toString();
    }
    
    // UI Navigation Methods
    showLoginScreen() {
        this.loginScreen.classList.remove('hidden');
        this.appScreen.classList.add('hidden');
        this.showLogin();
    }
    
    showApp() {
        if (this.currentUser) {
            this.welcomeUser.textContent = `Welcome, ${this.currentUser.fullname || this.currentUser.username}`;
            this.loginScreen.classList.add('hidden');
            this.appScreen.classList.remove('hidden');
            
            // Start camera
            this.camera.startCamera();
            
            // Reset any previous state
            this.resetRecordingUI();
        }
    }
    
    showLogin() {
        this.registerContainer.classList.add('hidden');
        this.loginContainer.classList.remove('hidden');
        this.usernameInput.focus();
    }
    
    showRegister() {
        this.loginContainer.classList.add('hidden');
        this.registerContainer.classList.remove('hidden');
        this.regNameInput.focus();
    }
    
    showModal(title, message, callback = null) {
        this.modalTitle.textContent = title;
        this.modalMessage.textContent = message;
        this.modal.classList.remove('hidden');
        
        // Store callback if provided
        this.modalConfirm.onclick = () => {
            this.hideModal();
            if (callback) callback();
        };
    }
    
    hideModal() {
        this.modal.classList.add('hidden');
    }
    
    // Recording Methods
    toggleRecording() {
        if (this.isRecording) return;
        
        this.isRecording = true;
        this.remainingTime = 60;
        
        // Update UI
        this.recordBtn.disabled = true;
        this.recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Recording...';
        document.querySelector('.recording-buttons').classList.remove('hidden');
        this.timerDisplay.classList.remove('hidden');
        this.updateTimerDisplay();
        
        // Start countdown
        this.timerInterval = setInterval(() => {
            this.remainingTime--;
            this.updateTimerDisplay();
            
            if (this.remainingTime <= 0) {
                clearInterval(this.timerInterval);
                this.finishRecording();
            }
        }, 1000);
        
        // Start recording
        this.recorder.startRecording();
    }
    
    togglePause() {
        if (!this.isRecording) return;
        
        if (this.isPaused) {
            // Resume recording
            this.isPaused = false;
            this.pauseBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
            this.recorder.resumeRecording();
            
            // Restart timer
            this.timerInterval = setInterval(() => {
                this.remainingTime--;
                this.updateTimerDisplay();
                
                if (this.remainingTime <= 0) {
                    clearInterval(this.timerInterval);
                    this.finishRecording();
                }
            }, 1000);
        } else {
            // Pause recording
            this.isPaused = true;
            this.pauseBtn.innerHTML = '<i class="fas fa-play"></i> Resume';
            this.recorder.pauseRecording();
            
            // Pause timer
            clearInterval(this.timerInterval);
        }
    }
    
    cancelRecording() {
        if (!this.isRecording) return;
        
        // Stop recording and timer
        this.recorder.stopRecording();
        clearInterval(this.timerInterval);
        
        // Reset state
        this.isRecording = false;
        this.isPaused = false;
        
        // Reset UI
        this.resetRecordingUI();
    }
    
    finishRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        this.isPaused = false;
        
        // Stop recording
        this.recorder.stopRecording();
        
        // Process recording
        this.processRecording();
    }
    
    async processRecording() {
        // Show processing message
        this.timerDisplay.textContent = "Processing...";
        
        try {
            // Get the recorded audio blob
            const audioBlob = this.recorder.getAudioBlob();
            
            if (!audioBlob) {
                throw new Error("No audio recorded");
            }
            
            // For demo purposes, we'll simulate transcription and analysis
            // In a real app, you would send the audio to a backend service for processing
            
            // Simulate transcription delay
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Get the selected question
            const question = this.questionDropdown.value;
            
            // For demo, use a pre-defined response
            const demoResponses = [
                "I'm a highly motivated professional with five years of experience in project management. I've successfully led several cross-functional teams to deliver projects on time and under budget.",
                "My greatest strength is my ability to communicate effectively with both technical and non-technical stakeholders. I'm also very organized and detail-oriented.",
                "Sometimes I tend to take on too much responsibility myself instead of delegating. I'm working on this by learning to trust my team members more.",
                "I'm excited about your company's innovative approach to solving customer problems. Your mission aligns perfectly with my own professional values.",
                "In five years, I see myself in a leadership role, mentoring others and contributing to strategic decisions.",
                "You should hire me because I bring both technical expertise and strong interpersonal skills. I can hit the ground running and add immediate value to your team.",
                "Last year, my team faced a major deadline crisis. I organized daily stand-ups, reprioritized tasks, and we delivered the project on time with positive feedback.",
                "My greatest achievement was implementing a new system that reduced processing time by 30%, saving the company over $100,000 annually.",
                "I handle stress by breaking problems into smaller tasks and maintaining open communication with my team. Regular exercise also helps me stay balanced.",
                "Yes, I'd love to hear more about the team dynamics and opportunities for professional development in this role."
            ];
            
            const responseIndex = this.questionDropdown.selectedIndex;
            const demoText = demoResponses[responseIndex] || demoResponses[0];
            
            // Display the question and "transcribed" text
            this.questionDisplay.textContent = `Question: ${question}`;
            this.transcriptionDisplay.textContent = `Your Response:\n${demoText}`;
            
            // Analyze the response
            this.currentAnalysis = this.analyzer.analyzeResponse(demoText);
            
            // Display analysis results
            this.displayAnalysisResults(this.currentAnalysis);
            
            // Enable download button
            this.downloadBtn.classList.remove('hidden');
            
        } catch (error) {
            console.error("Error processing recording:", error);
            this.showModal('Error', 'Failed to process recording. Please try again.');
        }
        
        // Reset recording UI
        this.resetRecordingUI();
    }
    
    resetRecordingUI() {
        this.recordBtn.disabled = false;
        this.recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record Answer (60 sec)';
        document.querySelector('.recording-buttons').classList.add('hidden');
        this.timerDisplay.classList.add('hidden');
        this.timerDisplay.textContent = "00:60";
        this.isRecording = false;
        this.isPaused = false;
        this.remainingTime = 60;
        clearInterval(this.timerInterval);
    }
    
    updateTimerDisplay() {
        const minutes = Math.floor(this.remainingTime / 60);
        const seconds = this.remainingTime % 60;
        this.timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    // Analysis Display Methods
    displayAnalysisResults(analysis) {
        this.analysisResults.innerHTML = '';
        
        // Tone Analysis
        const toneSection = document.createElement('div');
        toneSection.className = 'analysis-section';
        toneSection.innerHTML = `
            <h3>Tone Analysis</h3>
            <p><strong>Sentiment:</strong> ${analysis.tone.sentiment} (Score: ${analysis.tone.score.toFixed(2)})</p>
            <p>${analysis.tone.feedback}</p>
        `;
        this.analysisResults.appendChild(toneSection);
        
        // Word Choice Analysis
        const wordChoiceSection = document.createElement('div');
        wordChoiceSection.className = 'analysis-section';
        wordChoiceSection.innerHTML = `
            <h3>Word Choice Analysis</h3>
            <p><strong>Professional words used:</strong> ${analysis.wordChoice.professionalWordCount}</p>
            ${analysis.wordChoice.professionalWordsUsed.length > 0 ? 
                `<p><strong>Examples:</strong> ${analysis.wordChoice.professionalWordsUsed.slice(0, 5).join(', ')}</p>` : ''}
            <p><strong>Filler words/phrases used:</strong> ${analysis.wordChoice.fillerWordCount}</p>
            ${analysis.wordChoice.fillerWordsUsed.length > 0 ? 
                `<p><strong>Examples:</strong> ${analysis.wordChoice.fillerWordsUsed.slice(0, 5).join(', ')}</p>` : ''}
            <p>${analysis.wordChoice.feedback}</p>
        `;
        this.analysisResults.appendChild(wordChoiceSection);
        
        // Confidence Analysis
        const confidenceSection = document.createElement('div');
        confidenceSection.className = 'analysis-section';
        confidenceSection.innerHTML = `
            <h3>Confidence Assessment</h3>
            <p><strong>Confidence Score:</strong> ${analysis.confidence.confidenceScore.toFixed(1)}/10</p>
            <p>${analysis.confidence.feedback}</p>
        `;
        this.analysisResults.appendChild(confidenceSection);
        
        // Overall Feedback
        const overallSection = document.createElement('div');
        overallSection.className = 'analysis-section';
        overallSection.innerHTML = `
            <h3>Overall Feedback</h3>
            <p>${analysis.overallFeedback}</p>
            <h4>Areas to Focus On:</h4>
            <ul>
                ${analysis.improvementAreas.map(area => `<li>${area}</li>`).join('')}
            </ul>
        `;
        this.analysisResults.appendChild(overallSection);
    }
    
    downloadReport() {
        if (!this.currentAnalysis) return;
    
        // Get current date for filename
        const now = new Date();
        const dateStr = now.toISOString().split('T')[0];
        
        // Create report content
        let reportContent = `Interview Coach Analysis Report - ${dateStr}\n\n`;
        reportContent += `Candidate: ${this.currentUser.fullname || this.currentUser.username}\n`;
        reportContent += `Question: ${this.questionDropdown.value}\n\n`;
        reportContent += `Response Transcript:\n${this.transcriptionDisplay.textContent}\n\n`;
        reportContent += "Analysis Results:\n";
        reportContent += "-----------------\n\n";
        
        // Tone Analysis
        reportContent += "Tone Analysis:\n";
        reportContent += `Sentiment: ${this.currentAnalysis.tone.sentiment} (Score: ${this.currentAnalysis.tone.score.toFixed(2)})\n`;
        reportContent += `${this.currentAnalysis.tone.feedback}\n\n`;
        
        // Word Choice Analysis
        reportContent += "Word Choice Analysis:\n";
        reportContent += `Professional Words Used: ${this.currentAnalysis.wordChoice.professionalWordCount}\n`;
        if (this.currentAnalysis.wordChoice.professionalWordsUsed.length > 0) {
            reportContent += `Examples: ${this.currentAnalysis.wordChoice.professionalWordsUsed.slice(0, 5).join(', ')}\n`;
        }
        reportContent += `Filler Words Used: ${this.currentAnalysis.wordChoice.fillerWordCount}\n`;
        if (this.currentAnalysis.wordChoice.fillerWordsUsed.length > 0) {
            reportContent += `Examples: ${this.currentAnalysis.wordChoice.fillerWordsUsed.slice(0, 5).join(', ')}\n`;
        }
        reportContent += `${this.currentAnalysis.wordChoice.feedback}\n\n`;
        
        // Confidence Assessment
        reportContent += "Confidence Assessment:\n";
        reportContent += `Score: ${this.currentAnalysis.confidence.confidenceScore.toFixed(1)}/10\n`;
        reportContent += `${this.currentAnalysis.confidence.feedback}\n\n`;
        
        // Overall Feedback
        reportContent += "Overall Feedback:\n";
        reportContent += `${this.currentAnalysis.overallFeedback}\n\n`;
        reportContent += "Areas for Improvement:\n";  // Fixed line (removed stray backtick)
        this.currentAnalysis.improvementAreas.forEach((area, index) => {
            reportContent += `${index + 1}. ${area}\n`;
        });
        
        // Create Blob and download
        try {
            const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `InterviewCoach-Analysis-${this.currentUser.username}-${dateStr}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error generating report:', error);
            this.showModal('Download Error', 'Failed to generate report. Please try again.');
        }
    }
}