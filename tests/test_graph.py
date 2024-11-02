import unittest
import networkx as nx
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.graph.graph_processor import GraphProcessor

class TestGraphProcessor(unittest.TestCase):
    def setUp(self):
        self.graph_processor = GraphProcessor(config_path="../configs/config.yaml",
                                              graph_config_path="../configs/graph_config.yaml")

    def test_load_emotion_patterns(self):
        self.graph_processor._load_emotion_patterns()

        # Check if specific emotions were loaded correctly
        self.assertIn("afraid", self.graph_processor.emotion_graph.nodes)
        afraid_data = self.graph_processor.emotion_graph.nodes["afraid"]
        self.assertIn("common_causes", afraid_data)
        self.assertIn("related_emotions", afraid_data)
        self.assertIn("response_patterns", afraid_data)

        self.assertIn("angry", self.graph_processor.emotion_graph.nodes)
        angry_data = self.graph_processor.emotion_graph.nodes["angry"]
        self.assertIn("common_causes", angry_data)
        self.assertIn("related_emotions", angry_data)
        self.assertIn("response_patterns", angry_data)

        self.assertIn("sad", self.graph_processor.emotion_graph.nodes)
        sad_data = self.graph_processor.emotion_graph.nodes["sad"]
        self.assertIn("common_causes", sad_data)
        self.assertIn("related_emotions", sad_data)
        self.assertIn("response_patterns", sad_data)

    def test_process_emotion(self):
        # Test processing a new emotion
        insights = self.graph_processor.process_emotion("happy", "good_news", "Received a job offer")
        self.assertEqual(insights["current_emotion"], "happy")
        self.assertGreaterEqual(len(insights["related_emotions"]), 0)
        self.assertGreater(len(insights["response_patterns"]), 0)
        self.assertGreater(len(insights["common_causes"]), 0)

        # Test processing an unknown emotion
        insights = self.graph_processor.process_emotion("unknown_emotion", "something", "unknown context")
        self.assertEqual(insights["current_emotion"], "neutral")

    def test_update_preference(self):
        self.graph_processor.process_emotion("happy", "good_news", "Received a job offer")
        self.graph_processor.update_preference("happy", {"style": "default"}, [{"style": "alternative"}])
        self.assertIn("happy", self.graph_processor.preference_history)
        self.assertIn("default", self.graph_processor.preference_history["happy"]["styles"])
        self.assertIn("alternative", self.graph_processor.preference_history["happy"]["styles"])

    def test_save_and_load_graph(self):
        # Process some emotions
        self.graph_processor.process_emotion("happy", "good_news", "Received a job offer")
        self.graph_processor.process_emotion("sad", "personal_loss", "Grandparent passed away")

        # Save the graph
        self.graph_processor.save_graph()

        # Load the graph
        self.graph_processor.load_graph()

        # Check if the loaded graph has the expected nodes and edges
        self.assertGreater(len(self.graph_processor.emotion_graph.nodes), 0)
        self.assertGreater(len(self.graph_processor.emotion_graph.edges), 0)
        self.assertIn("happy", self.graph_processor.emotion_graph.nodes)
        self.assertIn("sad", self.graph_processor.emotion_graph.nodes)

    def test_process_emotion_inference(self):
        # Test processing a known emotion
        insights = self.graph_processor.process_emotion("afraid", "dangerous_situation", "Dark alley at night")
        self.assertEqual(insights["current_emotion"], "afraid")
        self.assertGreaterEqual(len(insights["related_emotions"]), 0)
        self.assertGreater(len(insights["response_patterns"]), 0)
        self.assertGreater(len(insights["common_causes"]), 0)

        # Check if the related emotions make sense
        related_emotions = [emotion["emotion"] for emotion in insights["related_emotions"]]
        if related_emotions:
            self.assertIn("terrified", related_emotions)
            self.assertIn("anxious", related_emotions)
            self.assertIn("apprehensive", related_emotions)
            self.assertIn("worried", related_emotions)
            self.assertIn("nervous", related_emotions)

        # Check if the response patterns make sense
        response_patterns = insights["response_patterns"]
        self.assertIn("Acknowledge their fear and show understanding", response_patterns)
        self.assertIn("Offer reassurance and support", response_patterns)
        self.assertIn("Help assess the real threat level", response_patterns)
        self.assertIn("Share safety strategies", response_patterns)
        self.assertIn("Stay present with them", response_patterns)

        # Test processing an unknown emotion
        unknown_insights = self.graph_processor.process_emotion("unknown_emotion", "something", "unknown context")
        self.assertEqual(unknown_insights["current_emotion"], "neutral")
        self.assertEqual(unknown_insights["related_emotions"], [])
        self.assertEqual(unknown_insights["response_patterns"], [
            "Show understanding",
            "Offer support",
            "Share perspective"
        ])
        self.assertEqual(unknown_insights["common_causes"], ["something"])

    def test_visualize_emotion_graph(self):
        self.graph_processor._load_emotion_patterns()
        self.graph_processor.visualize_emotion_graph()

if __name__ == "__main__":
    unittest.main()