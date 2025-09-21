class AudioRecorder {
    constructor(app) {
        this.app = app;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.isRecording = false;
        this.isPaused = false;
    }

    async startRecording() {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            // Setup event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                // Create audio blob from chunks
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                
                // Stop all tracks in the stream
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            this.isPaused = false;
            
        } catch (error) {
            console.error("Error accessing microphone:", error);
            this.app.showModal('Error', 'Could not access microphone. Please ensure you have granted permission.');
            this.app.cancelRecording();
        }
    }

    pauseRecording() {
        if (this.mediaRecorder && this.isRecording && !this.isPaused) {
            this.mediaRecorder.pause();
            this.isPaused = true;
        }
    }

    resumeRecording() {
        if (this.mediaRecorder && this.isRecording && this.isPaused) {
            this.mediaRecorder.resume();
            this.isPaused = false;
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.isPaused = false;
        }
    }

    getAudioBlob() {
        return this.audioBlob;
    }

    // For demo purposes - in a real app you would send this to a backend for processing
    async transcribeAudio() {
        if (!this.audioBlob) {
            throw new Error("No audio recorded");
        }
        
        // In a real implementation, you would:
        // 1. Send the audioBlob to your backend
        // 2. Backend would use a speech-to-text API
        // 3. Return the transcription
        
        // For demo, we'll simulate this with a timeout
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Return a dummy transcription based on the selected question
        const questions = [
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
        ];
        
        const responses = [
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
        
        const questionIndex = questions.indexOf(this.app.questionDropdown.value);
        return questionIndex >= 0 ? responses[questionIndex] : responses[0];
    }
}