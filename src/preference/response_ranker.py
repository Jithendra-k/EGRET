from typing import Dict, List
import json
from pathlib import Path
import logging
from datetime import datetime


class ResponseRanker:
    def __init__(self, preference_file: str = "data/preference_history/preferences.json"):
        self.preference_file = Path(preference_file)
        self.logger = logging.getLogger(__name__)
        self.preferences = self._load_preferences()

    def save_preference(self, context: Dict, selected_response: Dict, responses: List[Dict]):
        """Save user's response preference with context"""
        try:
            preference_data = {
                'timestamp': datetime.now().isoformat(),
                'context': {
                    'message': context['message'],
                    'emotion': context['emotion_data']['emotion'],
                    'cause': context['emotion_data']['cause']
                },
                'selected': selected_response,
                'alternatives': [r for r in responses if r['text'] != selected_response['text']]
            }

            self.preferences.append(preference_data)
            self._save_preferences()
            self.logger.info(f"Saved preference for emotion: {context['emotion_data']['emotion']}")

        except Exception as e:
            self.logger.error(f"Error saving preference: {str(e)}")

    def _load_preferences(self) -> List[Dict]:
        """Load existing preferences with error handling"""
        try:
            if self.preference_file.exists():
                with open(self.preference_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # Handle empty file
                        self.logger.info("Empty preferences file found, initializing new preferences")
                        return []
                    return json.loads(content)
            else:
                self.logger.info("No preferences file found, initializing new preferences")
                return []

        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON in preferences file: {str(e)}. Starting with empty preferences")
            return []
        except Exception as e:
            self.logger.error(f"Error loading preferences: {str(e)}")
            return []

    def _save_preferences(self):
        """Save preferences with error handling"""
        try:
            self.preference_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.preference_file, 'w') as f:
                json.dump(self.preferences, f, indent=2, default=str)
            self.logger.debug(f"Successfully saved {len(self.preferences)} preferences")

        except Exception as e:
            self.logger.error(f"Error saving preferences file: {str(e)}")

    def get_preference_stats(self) -> Dict:
        """Get statistics about saved preferences"""
        try:
            total_preferences = len(self.preferences)
            emotions = [p['context']['emotion'] for p in self.preferences]
            emotion_counts = {e: emotions.count(e) for e in set(emotions)}

            return {
                'total_preferences': total_preferences,
                'emotion_distribution': emotion_counts,
                'last_updated': self.preferences[-1]['timestamp'] if self.preferences else None
            }
        except Exception as e:
            self.logger.error(f"Error getting preference stats: {str(e)}")
            return {
                'total_preferences': 0,
                'emotion_distribution': {},
                'last_updated': None
            }