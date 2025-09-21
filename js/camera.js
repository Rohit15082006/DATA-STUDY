class CameraHandler {
    constructor(app) {
        this.app = app;
        this.videoElement = document.getElementById('webcam');
        this.canvasElement = document.getElementById('posture-canvas');
        this.canvasContext = this.canvasElement.getContext('2d');
        this.stream = null;
        this.faceDetector = null;
        this.facePositions = [];
        this.isRunning = false;
        this.animationFrameId = null;
        
        // Initialize face detector
        this.initFaceDetector();
    }

    async initFaceDetector() {
        // Load face detection model
        // Note: In a production app, you might want to use a more sophisticated model
        // like TensorFlow.js or a dedicated face detection API
        
        // For this demo, we'll use a simple approach that tracks face position
        // based on color contrast (not real face detection)
        this.faceDetector = {
            detect: async () => {
                // This would be replaced with actual face detection logic
                // For demo purposes, we'll return a mock face position
                return [{
                    x: this.canvasElement.width * 0.4,
                    y: this.canvasElement.height * 0.4,
                    width: this.canvasElement.width * 0.2,
                    height: this.canvasElement.height * 0.3
                }];
            }
        };
    }

    async startCamera() {
        if (this.isRunning) return;
        
        try {
            // Get video stream
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            // Set video source
            this.videoElement.srcObject = this.stream;
            this.videoElement.play();
            
            // Set canvas dimensions
            this.canvasElement.width = this.videoElement.offsetWidth;
            this.canvasElement.height = this.videoElement.offsetHeight;
            
            // Start processing frames
            this.isRunning = true;
            this.processFrame();
            
        } catch (error) {
            console.error("Error accessing camera:", error);
            this.app.showModal('Error', 'Could not access camera. Please ensure you have granted permission.');
        }
    }

    stopCamera() {
        this.isRunning = false;
        
        // Stop video stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Stop any animation frame loop
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        // Clear canvas
        this.canvasContext.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }

    async processFrame() {
        if (!this.isRunning) return;
        
        try {
            // Draw video frame to canvas
            this.canvasContext.drawImage(
                this.videoElement,
                0, 0,
                this.canvasElement.width,
                this.canvasElement.height
            );
            
            // Detect faces (mock detection in this demo)
            const faces = await this.faceDetector.detect();
            
            if (faces && faces.length > 0) {
                // Draw face bounding box
                faces.forEach(face => {
                    this.canvasContext.strokeStyle = '#FF0000';
                    this.canvasContext.lineWidth = 2;
                    this.canvasContext.strokeRect(
                        face.x, face.y,
                        face.width, face.height
                    );
                    
                    // Track face position for posture analysis
                    const centerY = face.y + face.height / 2;
                    this.facePositions.push(centerY);
                    
                    // Keep only the last 10 positions
                    if (this.facePositions.length > 10) {
                        this.facePositions.shift();
                    }
                });
            }
            
            // Update posture assessment
            this.updatePostureAssessment();
            
        } catch (error) {
            console.error("Error processing frame:", error);
        }
        
        // Continue processing frames
        this.animationFrameId = requestAnimationFrame(() => this.processFrame());
    }

    updatePostureAssessment() {
        if (this.facePositions.length === 0) {
            this.app.postureLabel.textContent = "Posture Assessment: No face detected";
            return;
        }
        
        // Calculate variance in face position
        const mean = this.facePositions.reduce((sum, pos) => sum + pos, 0) / this.facePositions.length;
        const variance = this.facePositions.reduce((sum, pos) => sum + Math.pow(pos - mean, 2), 0) / this.facePositions.length;
        
        // Calculate average position
        const avgPosition = mean / this.canvasElement.height;
        
        let assessment;
        if (variance > 100) {
            assessment = "Poor - Too much movement";
        } else if (avgPosition > 0.6) {
            assessment = "Poor - Head position too low";
        } else if (avgPosition < 0.3) {
            assessment = "Check camera position";
        } else {
            assessment = "Good posture";
        }
        
        this.app.postureLabel.textContent = `Posture Assessment: ${assessment}`;
    }
}