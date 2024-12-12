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

    def preference_learning(self, selected_response: Dict, all_responses: List[Dict]) -> None:
        """
        Learn from user's response selection using epsilon-greedy strategy for exploration/exploitation.

        Args:
            selected_response (Dict): The response chosen by user
            all_responses (List[Dict]): List of all generated responses including the selected one
        """
        try:
            # Initialize if first time
            if not hasattr(self, 'learning_history'):
                self.learning_history = []
                self.feature_weights = {
                    'text_length': 1.0,
                    'has_emoji': 1.0,
                    'has_question': 1.0,
                    'sentiment_words': 1.0
                }
                self.epsilon = 0.2  # Initial exploration rate
                self.min_epsilon = 0.05  # Minimum exploration rate
                self.epsilon_decay = 0.995  # Decay factor
                self.selection_counts = {}  # Track selection frequency

            current_time = datetime.now().isoformat()

            # Exploration vs Exploitation decision
            is_exploring = random.random() < self.epsilon

            if is_exploring:
                # Exploration: Random feature focus
                focus_feature = random.choice(list(self.feature_weights.keys()))
                self.feature_weights[focus_feature] *= 1.2  # Temporary boost for exploration
                self.logger.info(f"Exploring: Focusing on feature {focus_feature}")
            else:
                # Exploitation: Use current best weights
                self.logger.info("Exploiting: Using current best weights")

            # Extract and analyze features
            selected_features = {
                'text_length': len(selected_response['text']),
                'has_emoji': any(char in selected_response['text'] for char in ['😊', '🙂', '😃', '❤️', '👍']),
                'has_question': '?' in selected_response['text'],
                'sentiment_words': sum(1 for word in ['happy', 'sad', 'excited', 'worried', 'glad']
                                       if word in selected_response['text'].lower())
            }

            # Track selection
            response_key = f"{selected_response['text'][:20]}..."  # Truncated text as key
            self.selection_counts[response_key] = self.selection_counts.get(response_key, 0) + 1

            # Record learning data with exploration flag
            selection_data = {
                'timestamp': current_time,
                'selected_features': selected_features,
                'position_in_list': next((i for i, r in enumerate(all_responses)
                                          if r['text'] == selected_response['text']), -1),
                'total_options': len(all_responses),
                'was_exploring': is_exploring,
                'epsilon': self.epsilon
            }
            self.learning_history.append(selection_data)

            # Update weights based on selection and exploration status
            learning_rate = 0.2 if is_exploring else 0.1  # Higher learning rate during exploration
            for feature, value in selected_features.items():
                if value:  # If feature is present in selected response
                    self.feature_weights[feature] *= (1 + learning_rate)
                else:
                    self.feature_weights[feature] *= (1 - learning_rate * 0.5)  # Smaller penalty

            # Normalize weights
            total_weight = sum(self.feature_weights.values())
            self.feature_weights = {k: v / total_weight for k, v in self.feature_weights.items()}

            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Log learning progress
            self.logger.info(f"Exploration rate (epsilon): {self.epsilon:.3f}")
            self.logger.info(f"Updated weights: {self.feature_weights}")
            if len(self.selection_counts) > 0:
                most_selected = max(self.selection_counts.items(), key=lambda x: x[1])
                self.logger.info(f"Most selected response pattern: {most_selected[0]} ({most_selected[1]} times)")

            # Save learning state
            self._save_preferences()

        except Exception as e:
            self.logger.error(f"Error in learning from selection: {str(e)}")