from typing import Dict, List
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationEntry:
    user_message: str
    emotion: Dict
    system_response: str
    timestamp: float


class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.history: List[ConversationEntry] = []
        self.max_history = max_history

    def add_exchange(self, user_msg: str, emotion: Dict, system_response: str) -> None:
        entry = ConversationEntry(
            user_message=user_msg,
            emotion=emotion,
            system_response=system_response,
            timestamp=time.time()
        )
        self.history.append(entry)
        self.history = self.history[-self.max_history:]

    def get_formatted_history(self) -> str:
        if not self.history:
            return "No previous conversation."

        formatted = []
        for entry in self.history:
            timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M')
            formatted.extend([
                f"[{timestamp}] User: {entry.user_message}",
                f"Emotion: {entry.emotion['emotion']} due to {entry.emotion['cause']}",
                f"Assistant: {entry.system_response}\n"
            ])
        return "\n".join(formatted)

    def clear_history(self) -> None:
        self.history.clear()

    def get_emotional_context(self) -> List[Dict]:
        return [
            {
                'emotion': entry.emotion['emotion'],
                'cause': entry.emotion['cause'],
                'timestamp': entry.timestamp
            }
            for entry in self.history
        ]