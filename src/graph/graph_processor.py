import networkx as nx
import json
from pathlib import Path
import logging
from typing import Dict, List
import yaml
from datetime import datetime
import matplotlib.pyplot as plt


class GraphProcessor:
    def __init__(self, config_path: str = "configs/config.yaml",
                 graph_config_path: str = "configs/graph_config.yaml"):
        # First load main config
        self.config = self._load_config(config_path)

        # Setup logging before anything else
        logging.basicConfig(level=self.config['system']['log_level'])
        self.logger = logging.getLogger(__name__)

        # Now load graph config
        try:
            self.graph_config = self._load_graph_config(graph_config_path)
        except Exception as e:
            self.logger.error(f"Error loading graph config: {str(e)}")
            raise

        # Initialize graph
        self.emotion_graph = nx.DiGraph()

        # Initialize counters
        self.message_counter = 0
        self.interaction_counter = 0
        self.preference_history = {}

        # Load initial patterns
        self._load_emotion_patterns()

        # Setup storage
        self.save_dir = Path("data/emotion_patterns")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load main configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_graph_config(self, config_path: str) -> Dict:
        """Load graph-specific configuration"""
        with open(config_path, 'r') as f:
            graph_config = yaml.safe_load(f)
            self.logger.info("Loaded graph configuration")
            return graph_config

    def _load_emotion_patterns(self):
        """Load base emotion patterns"""
        try:
            patterns_path = Path("data/emotion_patterns/base_patterns.json")
            if patterns_path.exists():
                with open(patterns_path, 'r') as f:
                    patterns = json.load(f)
                    self._initialize_graph(patterns)
                    self.logger.info("Loaded base emotion patterns")
            else:
                self.logger.warning("No base patterns found. Starting with empty graph")
        except Exception as e:
            self.logger.error(f"Error loading emotion patterns: {str(e)}")
            raise

    def _initialize_graph(self, patterns: Dict):
        """Initialize graph with emotion patterns"""
        for emotion, data in patterns.items():
            # Add emotion node
            self.emotion_graph.add_node(
                emotion,
                type='emotion',
                common_causes=data['common_causes'],
                related_emotions=data.get('related_emotions', []),
                response_patterns=data['response_patterns']
            )

            # Add relationships to related emotions
            for related in data.get('related_emotions', []):
                self.emotion_graph.add_edge(
                    emotion,
                    related,
                    weight=self.graph_config['graph_structure']['initial_weight'],
                    interaction_count=0
                )

    def visualize_emotion_processing(self, current_emotion: str, insights: Dict):
        """Create a simplified visualization of emotion processing"""
        plt.figure(figsize=(12, 8))

        # Create a new graph for visualization
        viz_graph = nx.DiGraph()

        # Add current emotion as central node
        viz_graph.add_node(current_emotion, node_type='current', size=2000)

        # Add related emotions
        for related in insights['related_emotions']:
            emotion_name = related['emotion']
            strength = related['strength']
            viz_graph.add_node(emotion_name, node_type='related', size=1000)
            viz_graph.add_edge(current_emotion, emotion_name, weight=strength)

        # Add response patterns as nodes
        for i, pattern in enumerate(insights['response_patterns']):
            pattern_name = f"Pattern {i + 1}\n{pattern}"
            viz_graph.add_node(pattern_name, node_type='pattern', size=1500)
            viz_graph.add_edge(current_emotion, pattern_name, style='dashed')

        # Position nodes
        pos = nx.spring_layout(viz_graph, k=1, iterations=50)

        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in viz_graph.nodes():
            node_type = viz_graph.nodes[node]['node_type']
            if node_type == 'current':
                node_colors.append('#ff7f0e')  # Orange for current emotion
                node_sizes.append(2000)
            elif node_type == 'related':
                node_colors.append('#1f77b4')  # Blue for related emotions
                node_sizes.append(1000)
            else:
                node_colors.append('#2ca02c')  # Green for response patterns
                node_sizes.append(1500)

        # Draw edges with varying width based on weights
        edges = viz_graph.edges()
        edge_styles = [('solid' if 'weight' in viz_graph[u][v] else 'dashed') for u, v in edges]
        edge_weights = [viz_graph[u][v].get('weight', 1.0) for u, v in edges]

        # Draw the graph
        nx.draw_networkx_nodes(viz_graph, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(viz_graph, pos, edge_color='gray', style=edge_styles,
                               width=[w * 2 for w in edge_weights], alpha=0.5)
        nx.draw_networkx_labels(viz_graph, pos, font_size=10, font_weight='bold')

        plt.title(f"Emotion Processing Map for '{current_emotion}'")
        plt.axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Current Emotion',
                       markerfacecolor='#ff7f0e', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Related Emotions',
                       markerfacecolor='#1f77b4', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Response Patterns',
                       markerfacecolor='#2ca02c', markersize=12)
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return plt.gcf()

    def process_emotion(self, emotion: str, cause: str, context: str) -> Dict:
        """Modified process_emotion to include visualization"""
        try:
            # Original processing
            if emotion not in self.emotion_graph:
                self.logger.warning(f"Unknown emotion: {emotion}, using neutral")
                emotion = 'neutral'

            timestamp = datetime.now().timestamp()
            instance_id = f"{emotion}_{timestamp}"

            # Add instance node
            self.emotion_graph.add_node(
                instance_id,
                type='instance',
                emotion=emotion,
                cause=cause,
                context=context,
                timestamp=timestamp
            )
            print("=======================================================")
            print("processing emotion in graph:", emotion)
            # Process as before...
            self.emotion_graph.add_edge(instance_id, emotion, type='instance_of')
            emotion_node = self.emotion_graph.nodes[emotion]

            if cause not in emotion_node.get('common_causes', []):
                if 'common_causes' not in emotion_node:
                    emotion_node['common_causes'] = []
                emotion_node['common_causes'].append(cause)

            if 'response_patterns' not in emotion_node:
                emotion_node['response_patterns'] = [
                    "Show understanding",
                    "Offer support",
                    "Share perspective"
                ]

            self._update_relationships(instance_id)
            insights = self._generate_insights(instance_id)
            self.message_counter += 1

            # Create visualization
            fig = self.visualize_emotion_processing(emotion, insights)
            insights['visualization'] = fig

            return insights

        except Exception as e:
            self.logger.error(f"Error processing emotion: {str(e)}")
            return {
                'current_emotion': emotion,
                'related_emotions': [],
                'emotional_trajectory': [],
                'response_patterns': [
                    "Show understanding",
                    "Offer support",
                    "Share perspective"
                ],
                'common_causes': [cause] if cause else []
            }
    # def process_emotion(self, emotion: str, cause: str, context: str) -> Dict:
    #     try:
    #         # Validate emotion exists in graph
    #         if emotion not in self.emotion_graph:
    #             self.logger.warning(f"Unknown emotion: {emotion}, using neutral")
    #             emotion = 'neutral'
    #
    #         timestamp = datetime.now().timestamp()
    #         instance_id = f"{emotion}_{timestamp}"
    #
    #         # Add instance node
    #         self.emotion_graph.add_node(
    #             instance_id,
    #             type='instance',
    #             emotion=emotion,
    #             cause=cause,
    #             context=context,
    #             timestamp=timestamp
    #         )
    #
    #         # Connect to emotion type and update
    #         self.emotion_graph.add_edge(instance_id, emotion, type='instance_of')
    #         emotion_node = self.emotion_graph.nodes[emotion]
    #
    #         if cause not in emotion_node.get('common_causes', []):
    #             if 'common_causes' not in emotion_node:
    #                 emotion_node['common_causes'] = []
    #             emotion_node['common_causes'].append(cause)
    #
    #         # Ensure response patterns exist
    #         if 'response_patterns' not in emotion_node:
    #             emotion_node['response_patterns'] = [
    #                 "Show understanding",
    #                 "Offer support",
    #                 "Share perspective"
    #             ]
    #
    #         # Update relationships
    #         self._update_relationships(instance_id)
    #
    #         # Generate insights
    #         insights = self._generate_insights(instance_id)
    #
    #         # Increment counter
    #         self.message_counter += 1
    #
    #         return insights
    #
    #     except Exception as e:
    #         self.logger.error(f"Error processing emotion: {str(e)}")
    #         # Return default insights
    #         return {
    #             'current_emotion': emotion,
    #             'related_emotions': [],
    #             'emotional_trajectory': [],
    #             'response_patterns': [
    #                 "Show understanding",
    #                 "Offer support",
    #                 "Share perspective"
    #             ],
    #             'common_causes': [cause] if cause else []
    #         }

    def _generate_insights(self, instance_id: str) -> Dict:
        try:
            instance_data = self.emotion_graph.nodes[instance_id]
            current_emotion = instance_data['emotion']

            # Get emotion type data with defaults
            emotion_data = self.emotion_graph.nodes.get(current_emotion, {})

            # Get related emotions with weights (with error handling)
            related_emotions = []
            try:
                for _, related, data in self.emotion_graph.edges(current_emotion, data=True):
                    if data.get('interaction_count', 0) >= self.graph_config['relationships']['minimum_interactions']:
                        related_emotions.append({
                            'emotion': related,
                            'strength': data.get('weight', 1.0)
                        })
            except Exception as e:
                self.logger.warning(f"Error getting related emotions: {str(e)}")

            # Sort and limit related emotions
            related_emotions.sort(key=lambda x: x['strength'], reverse=True)
            related_emotions = related_emotions[:self.graph_config['relationships']['max_related_emotions']]

            # Get trajectory and patterns
            trajectory = self._get_emotional_trajectory(instance_id)
            response_patterns = emotion_data.get('response_patterns', [
                "Show understanding",
                "Offer support",
                "Share perspective"
            ])

            return {
                'current_emotion': current_emotion,
                'related_emotions': related_emotions,
                'emotional_trajectory': trajectory,
                'response_patterns': response_patterns,
                'common_causes': emotion_data.get('common_causes', [])
            }

        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return {
                'current_emotion': 'neutral',
                'related_emotions': [],
                'emotional_trajectory': [],
                'response_patterns': [
                    "Show understanding",
                    "Offer support",
                    "Share perspective"
                ],
                'common_causes': []
            }

    def _update_relationships(self, instance_id: str):
        """Update emotional relationships"""
        instance_data = self.emotion_graph.nodes[instance_id]
        current_emotion = instance_data['emotion']

        recent_instances = self._get_recent_instances(
            limit=self.graph_config['history']['recent_window']
        )

        for prev_id in recent_instances:
            if prev_id != instance_id:
                prev_data = self.emotion_graph.nodes[prev_id]
                prev_emotion = prev_data['emotion']

                if self.emotion_graph.has_edge(prev_emotion, current_emotion):
                    edge_data = self.emotion_graph[prev_emotion][current_emotion]
                    new_weight = edge_data['weight'] * (1 + self.graph_config['graph_structure']['weight_increment'])
                    edge_data['weight'] = min(
                        new_weight,
                        self.graph_config['graph_structure']['max_weight']
                    )
                    edge_data['interaction_count'] += 1
                else:
                    self.emotion_graph.add_edge(
                        prev_emotion,
                        current_emotion,
                        weight=self.graph_config['graph_structure']['initial_weight'],
                        interaction_count=1
                    )

    def _get_recent_instances(self, limit: int) -> List[str]:
        """Get recent emotion instances"""
        instances = [
            (node, data['timestamp'])
            for node, data in self.emotion_graph.nodes(data=True)
            if data.get('type') == 'instance'
        ]

        instances.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in instances[:limit]]

    def _get_emotional_trajectory(self, instance_id: str) -> List[Dict]:
        """Get recent emotional trajectory"""
        recent_instances = self._get_recent_instances(
            limit=self.graph_config['responses']['context_window']
        )

        trajectory = []
        for node in recent_instances:
            data = self.emotion_graph.nodes[node]
            trajectory.append({
                'emotion': data['emotion'],
                'cause': data['cause'],
                'timestamp': data['timestamp']
            })

        return trajectory

    def _get_response_patterns(self, emotion_data: Dict, emotion: str) -> List[str]:
        """Get response patterns considering preferences"""
        patterns = emotion_data['response_patterns']

        if emotion in self.preference_history:
            preferred_patterns = self.preference_history[emotion]['patterns']
            valid_patterns = [
                pattern for pattern in patterns
                if preferred_patterns.get(pattern, 0) >= self.graph_config['patterns']['min_pattern_strength']
            ]

            valid_patterns.sort(
                key=lambda x: preferred_patterns.get(x, 0),
                reverse=True
            )

            return valid_patterns[:self.graph_config['patterns']['max_patterns_per_emotion']]

        return patterns

    def update_preference(self, emotion: str, selected: Dict, rejected: List[Dict]):
        """Update preference history"""
        if emotion not in self.preference_history:
            self.preference_history[emotion] = {
                'patterns': {},
                'styles': {}
            }

        learning_rate = self.graph_config['optimization']['learning_rate']

        # Update selected response
        selected_style = selected.get('style', 'default')
        current_score = self.preference_history[emotion]['styles'].get(selected_style, 1.0)
        self.preference_history[emotion]['styles'][selected_style] = min(
            current_score * (1 + learning_rate),
            self.graph_config['graph_structure']['max_weight']
        )

        # Update rejected responses
        decay = self.graph_config['optimization']['decay_factor']
        for response in rejected:
            style = response.get('style', 'default')
            current_score = self.preference_history[emotion]['styles'].get(style, 1.0)
            self.preference_history[emotion]['styles'][style] = max(
                current_score * decay,
                self.graph_config['graph_structure']['min_weight']
            )

        self.interaction_counter += 1
        if self.should_save():
            self.save_graph()
            self.interaction_counter = 0

    def _clean_weak_relationships(self):
        """Remove weak relationships"""
        weak_edges = [
            (u, v) for u, v, d in self.emotion_graph.edges(data=True)
            if d['weight'] < self.graph_config['relationships']['weak_threshold']
                and d['interaction_count'] > self.graph_config['relationships']['minimum_interactions']
        ]

        self.emotion_graph.remove_edges_from(weak_edges)

    def _apply_weight_decay(self):
        """Apply decay to edge weights"""
        decay_factor = self.graph_config['graph_structure']['weight_decay']
        for u, v, d in self.emotion_graph.edges(data=True):
            d['weight'] = max(
                d['weight'] * decay_factor,
                self.graph_config['graph_structure']['min_weight']
            )

    def cleanup_history(self):
        """Cleanup old data"""
        if len(self.emotion_graph) > self.graph_config['history']['cleanup_threshold']:
            instances = [
                (node, data['timestamp'])
                for node, data in self.emotion_graph.nodes(data=True)
                if data.get('type') == 'instance'
            ]

            instances.sort(key=lambda x: x[1])
            to_remove = instances[:-self.graph_config['history']['max_instances']]

            for node, _ in to_remove:
                self.emotion_graph.remove_node(node)

    def should_save(self) -> bool:
        """Check if graph should be saved"""
        return (self.interaction_counter >=
                self.graph_config['storage']['save_frequency'])

    def save_graph(self):
        """Save current graph state"""
        try:
            # Save graph structure
            graph_data = nx.node_link_data(self.emotion_graph)
            graph_path = self.save_dir / "current_graph.json"

            with open(graph_path, 'w') as f:
                json.dump(graph_data, f, indent=2)

            # Save preference history
            pref_path = self.save_dir / "preference_history.json"
            with open(pref_path, 'w') as f:
                json.dump(self.preference_history, f, indent=2)

            self.message_counter = 0
            self.logger.info("Saved graph and preference history")

        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}")
            raise

    def load_graph(self):
        """Load saved graph state"""
        try:
            graph_path = self.save_dir / "current_graph.json"
            if graph_path.exists():
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                    self.emotion_graph = nx.node_link_graph(graph_data)

            pref_path = self.save_dir / "preference_history.json"
            if pref_path.exists():
                with open(pref_path, 'r') as f:
                    self.preference_history = json.load(f)

            self.logger.info("Loaded graph and preference history")

        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            raise

    def visualize_emotion_graph(self):
        """Visualize the emotion graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.emotion_graph)
        nx.draw(self.emotion_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
        plt.title("Emotion Graph")
        plt.show()