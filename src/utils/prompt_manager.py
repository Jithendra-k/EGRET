from typing import Dict, List
from string import Template


class PromptManager:
    """Manages prompt templates for the model"""

    def __init__(self):

        self.emotion_detection_template = Template("""Detect the emotion and its cause from this text. Provide your response in the following format:
emotion: [emotion name]
cause: [cause of the emotion]
Text: ${text}

Response:(you must provide the answer in the above format)""")

        self.response_generation_template = Template("""Given the following context, generate an empathetic response:

User Message: ${message}

Emotional Context:+
- Current Emotion: ${emotion}
- Cause: ${cause}

Historical Insights:
- Related Emotions: ${related_emotions}
- Common Response Patterns: ${response_patterns}
- Recent Emotional Journey: ${emotional_trajectory}

Generate a response that is empathetic and considering the emotional context. 
Remember to generate only response (it should be like a one line chat message, not like a paragraph) to use user message and dont give any explanation to it.
Response:""")

    def create_emotion_detection_prompt(self, text: str) -> str:
        """Create prompt for emotion detection"""
        return self.emotion_detection_template.substitute(
            text=text
        )

    def create_response_generation_prompt(self,
                                          message: str,
                                          emotion_data: Dict,
                                          graph_insights: Dict) -> str:
        """Create prompt for response generation"""
        # Format related emotions
        related_emotions = ", ".join([
            f"{e['emotion']} ({e['strength']:.2f})"
            for e in graph_insights['related_emotions']
        ])

        # Format response patterns
        response_patterns = ", ".join(graph_insights['response_patterns'])

        # Format emotional trajectory
        trajectory = " → ".join([
            f"{e['emotion']}"
            for e in graph_insights['emotional_trajectory']
        ])

        return self.response_generation_template.substitute(
            message=message,
            emotion=emotion_data['emotion'],
            cause=emotion_data['cause'],
            related_emotions=related_emotions,
            response_patterns=response_patterns,
            emotional_trajectory=trajectory
        )

    def format_response(self, response: str) -> str:
        """Format model's response for display"""
        # Clean up any artifacts from generation
        response = response.strip()
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        return response

    def create_system_prompt(self) -> str:
        """Create system prompt for model initialization"""
        return """You are an empathetic AI assistant who understands human emotions and provides supportive responses.
Your responses should be:
1. Emotionally aware and sensitive
2. Supportive and understanding
3. Appropriate to the emotional context
4. Natural and conversational"""