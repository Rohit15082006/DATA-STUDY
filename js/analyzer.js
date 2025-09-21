class ResponseAnalyzer {
    constructor(app) {
        this.app = app;
        
        // Professional vocabulary
        this.professionalWords = new Set([
            'accomplished', 'achieved', 'analyzed', 'coordinated', 'created',
            'delivered', 'developed', 'enhanced', 'executed', 'improved',
            'initiated', 'launched', 'managed', 'optimized', 'organized',
            'planned', 'resolved', 'spearheaded', 'streamlined', 'success'
        ]);
        
        // Filler words
        this.fillerWords = new Set([
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'so', 'well', 'just', 'stuff', 'things'
        ]);
        
        // Hesitation patterns
        this.hesitationPatterns = [
            /\bI think\b/i, /\bmaybe\b/i, /\bpossibly\b/i, /\bperhaps\b/i,
            /\bI guess\b/i, /\bsort of\b/i, /\bkind of\b/i, /\bI hope\b/i,
            /\bI'm not sure\b/i, /\bI don't know\b/i
        ];
    }

    analyzeResponse(text) {
        if (!text || text.trim().length === 0) {
            return {
                tone: {
                    score: 0,
                    sentiment: 'neutral',
                    feedback: 'No speech detected to analyze tone.'
                },
                wordChoice: {
                    professionalWordCount: 0,
                    fillerWordCount: 0,
                    professionalWordsUsed: [],
                    fillerWordsUsed: [],
                    feedback: 'No speech detected to analyze word choice.'
                },
                confidence: {
                    confidenceScore: 0,
                    feedback: 'No speech detected to analyze confidence.'
                },
                overallFeedback: 'No speech was detected in your response. Please try again.',
                improvementAreas: ['Ensure you speak clearly into the microphone']
            };
        }
        
        // Analyze tone
        const toneAnalysis = this.analyzeTone(text);
        
        // Analyze word choice
        const wordChoiceAnalysis = this.analyzeWordChoice(text);
        
        // Analyze confidence
        const confidenceAnalysis = this.analyzeConfidence(text, toneAnalysis.score);
        
        // Generate overall feedback
        const overallFeedback = this.generateOverallFeedback(toneAnalysis, wordChoiceAnalysis, confidenceAnalysis);
        
        // Identify improvement areas
        const improvementAreas = this.identifyImprovementAreas(toneAnalysis, wordChoiceAnalysis, confidenceAnalysis);
        
        return {
            tone: toneAnalysis,
            wordChoice: wordChoiceAnalysis,
            confidence: confidenceAnalysis,
            overallFeedback,
            improvementAreas
        };
    }

    analyzeTone(text) {
        // Simple sentiment analysis
        // In a real app, you would use a more sophisticated NLP library or API
        
        // Count positive and negative words
        const positiveWords = ['excited', 'passionate', 'enjoy', 'love', 'great', 'excellent', 
                             'success', 'achievement', 'improve', 'happy', 'pleased'];
        const negativeWords = ['problem', 'issue', 'difficult', 'hard', 'challenge', 
                             'stress', 'pressure', 'weakness', 'failure'];
        
        let positiveCount = 0;
        let negativeCount = 0;
        
        const words = text.toLowerCase().split(/\s+/);
        words.forEach(word => {
            if (positiveWords.includes(word)) positiveCount++;
            if (negativeWords.includes(word)) negativeCount++;
        });
        
        // Calculate sentiment score (-1 to 1)
        const totalWords = words.length;
        const score = totalWords > 0 ? (positiveCount - negativeCount) / totalWords : 0;
        
        // Determine sentiment
        let sentiment, feedback;
        if (score >= 0.1) {
            sentiment = 'positive';
            feedback = "Your tone is positive and enthusiastic, which is great for an interview. Keep up the energy!";
            if (score > 0.3) {
                feedback += " However, be careful not to come across as overly enthusiastic as it might seem insincere.";
            }
        } else if (score <= -0.1) {
            sentiment = 'negative';
            feedback = "Your tone comes across as somewhat negative. Try to use more positive language and emphasize your strengths and achievements.";
        } else {
            sentiment = 'neutral';
            feedback = "Your tone is neutral. While this is professional, try to inject some enthusiasm when discussing your achievements or interest in the role.";
        }
        
        return {
            score,
            sentiment,
            feedback
        };
    }

    analyzeWordChoice(text) {
        const words = text.toLowerCase().split(/\s+/);
        
        // Count professional words
        const professionalWordsUsed = words.filter(word => this.professionalWords.has(word));
        
        // Count filler words
        const fillerWordsUsed = [];
        this.fillerWords.forEach(filler => {
            if (text.toLowerCase().includes(filler)) {
                fillerWordsUsed.push(filler);
            }
        });
        
        // Generate feedback
        let feedback = "";
        if (professionalWordsUsed.length > 0) {
            feedback += `Good use of professional language! Words like ${professionalWordsUsed.slice(0, 3).join(', ')} strengthen your responses. `;
        } else {
            feedback += "Consider incorporating more professional language to highlight your skills and achievements. ";
        }
        
        if (fillerWordsUsed.length > 0) {
            feedback += `Try to reduce filler words/phrases like ${fillerWordsUsed.slice(0, 3).join(', ')}. These can make you sound less confident.`;
        } else {
            feedback += "You've done well avoiding filler words, which makes your speech sound more confident and prepared.";
        }
        
        return {
            professionalWordCount: professionalWordsUsed.length,
            fillerWordCount: fillerWordsUsed.length,
            professionalWordsUsed: [...new Set(professionalWordsUsed)], // Remove duplicates
            fillerWordsUsed: [...new Set(fillerWordsUsed)], // Remove duplicates
            feedback
        };
    }

    analyzeConfidence(text, toneScore) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const avgSentenceLength = sentences.reduce((sum, sentence) => {
            return sum + sentence.split(/\s+/).length;
        }, 0) / (sentences.length || 1);
        
        // Count hesitation patterns
        let hesitationCount = 0;
        this.hesitationPatterns.forEach(pattern => {
            hesitationCount += (text.match(pattern) || []).length;
        });
        
        // Calculate confidence score (0-10)
        let confidenceScore = 5; // Base score
        
        // Adjust based on tone
        confidenceScore += toneScore * 3;
        
        // Adjust based on hesitation
        confidenceScore -= hesitationCount * 0.5;
        
        // Adjust based on sentence length
        if (avgSentenceLength > 15) confidenceScore += 1;
        if (avgSentenceLength < 8) confidenceScore -= 1;
        
        // Clamp between 0 and 10
        confidenceScore = Math.max(0, Math.min(10, confidenceScore));
        
        // Generate feedback
        let feedback;
        if (confidenceScore >= 8) {
            feedback = "You sound very confident. Your delivery is strong and assertive.";
        } else if (confidenceScore >= 6) {
            feedback = "You sound reasonably confident. With a few adjustments, you could project even more authority.";
        } else if (confidenceScore >= 4) {
            feedback = "Your confidence level seems moderate. Try speaking more assertively and avoiding hesitant language.";
        } else {
            feedback = "You may want to work on projecting more confidence. Try reducing hesitant phrases and speaking with more conviction.";
        }
        
        return {
            confidenceScore,
            feedback
        };
    }

    generateOverallFeedback(tone, wordChoice, confidence) {
        const avgScore = ((tone.score + 1) * 5 + confidence.confidenceScore) / 2;
        
        let overallFeedback = "";
        
        if (avgScore >= 8) {
            overallFeedback = "Excellent interview response! You presented yourself very well.\n";
        } else if (avgScore >= 6) {
            overallFeedback = "Good interview response. With some minor improvements, you'll make an even stronger impression.\n";
        } else if (avgScore >= 4) {
            overallFeedback = "Acceptable interview response. Focus on the improvement areas mentioned above.\n";
        } else {
            overallFeedback = "Your interview response needs improvement. Consider practicing more with the suggestions provided.\n";
        }
        
        return overallFeedback;
    }

    identifyImprovementAreas(tone, wordChoice, confidence) {
        const areas = [];
        
        if (tone.score < 0) {
            areas.push("Using more positive language");
        }
        
        if (wordChoice.fillerWordCount > 3) {
            areas.push("Reducing filler words/phrases");
        }
        
        if (wordChoice.professionalWordCount < 2) {
            areas.push("Incorporating more professional vocabulary");
        }
        
        if (confidence.confidenceScore < 5) {
            areas.push("Building confidence in delivery");
        }
        
        if (areas.length === 0) {
            areas.push("Keep practicing to maintain your strong performance");
        }
        
        return areas;
    }
}