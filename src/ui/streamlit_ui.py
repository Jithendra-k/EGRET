import streamlit as st
from typing import Dict, List
# import time
# from pathlib import Path
# import json
# import networkx as nx
import math
import plotly.graph_objects as go
from collections import Counter


class StreamlitUI:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_emotion' not in st.session_state:
            st.session_state.current_emotion = None
        if 'response_options' not in st.session_state:
            st.session_state.response_options = []
        if 'show_debug' not in st.session_state:
            st.session_state.show_debug = False
        if 'processing' not in st.session_state:
            st.session_state.processing = False

    def display_header(self):
        """Display application header"""
        st.title("EGRET: Emotion Graph-Enhanced Response Generation with Transformative preference learning")
        st.markdown("""
        Share your thoughts and feelings, and I'll respond with understanding.
        """)

    def display_chat_history(self):
        """Display chat message history"""
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        if "emotion_data" in message and st.session_state.show_debug:
                            with st.expander("Emotional Context"):
                                st.json(message["emotion_data"])

            if st.session_state.processing:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st.empty()

    def display_response_options(self, responses: List[Dict]):
        """Display response options with selection"""
        st.session_state.response_options = responses

        if responses:
            response_container = st.container()
            with response_container:
                with st.chat_message("assistant"):
                    st.markdown("_Choose a response:_")
                    for idx, response in enumerate(responses):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"> {response['text']}")
                            if st.session_state.show_debug:
                                st.caption(f"Style: {response['style']}")
                        with col2:
                            if st.button("Select", key=f"resp_{idx}"):
                                self.select_response(idx)

    def select_response(self, index: int):
        """Handle response selection"""
        selected = st.session_state.response_options[index]
        other_responses = [
            r for i, r in enumerate(st.session_state.response_options)
            if i != index
        ]

        st.session_state.messages.append({
            "role": "assistant",
            "content": selected['text'],
            "emotion_data": st.session_state.current_emotion
        })

        self.model_handler.graph_processor.update_preference(
            emotion=st.session_state.current_emotion['emotion'],
            selected=selected,
            rejected=other_responses
        )

        st.session_state.response_options = []
        st.session_state.current_emotion = None
        st.session_state.processing = False
        st.rerun()

    def visualize_emotion_processing(self, result: Dict):
        """Create visualization of emotion processing with history"""
        fig = go.Figure()

        # Get current and historical emotions
        current_emotion = result['emotion_data']['emotion']
        historical_emotions = [
                                  msg["emotion_data"]["emotion"]
                                  for msg in st.session_state.messages
                                  if "emotion_data" in msg
                              ] + [current_emotion]

        # Create nodes for each unique emotion
        unique_emotions = list(set(historical_emotions))
        positions = {}

        # Calculate positions in a circle
        for i, emotion in enumerate(unique_emotions):
            angle = (2 * 3.14159 * i) / len(unique_emotions)
            positions[emotion] = (math.cos(angle), math.sin(angle))

        # Add edges (connections between consecutive emotions)
        edge_x = []
        edge_y = []
        for i in range(len(historical_emotions) - 1):
            x0, y0 = positions[historical_emotions[i]]
            x1, y1 = positions[historical_emotions[i + 1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Add edges to plot
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            mode='lines',
            name='Transitions'
        ))

        # Add nodes for emotions
        for emotion in unique_emotions:
            x, y = positions[emotion]
            count = historical_emotions.count(emotion)
            size = 20 + (count * 10)
            color = '#FF6B6B' if emotion == current_emotion else '#4ECDC4'
            opacity = 1.0 if emotion == current_emotion else 0.7

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=2, color='white')
                ),
                text=[emotion],
                textposition="bottom center",
                name=emotion,
                hovertemplate=f'Emotion: {emotion}<br>Occurrences: {count}'
            ))

        # Update layout
        fig.update_layout(
            title="Emotional Journey Map",
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            height=500,
            width=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2])
        )

        return fig

    def display_emotion_timeline(self):
        """Create timeline visualization of emotions"""
        if not st.session_state.messages:
            st.info("Start a conversation to see emotional patterns")
            return

        emotions = [
            {"emotion": msg["emotion_data"]["emotion"],
             "cause": msg["emotion_data"]["cause"],
             "time": i}
            for i, msg in enumerate(st.session_state.messages)
            if "emotion_data" in msg
        ]

        if emotions:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[e["time"] for e in emotions],
                y=[e["emotion"] for e in emotions],
                mode='lines+markers',
                name='Emotion Timeline',
                hovertemplate='Emotion: %{y}<br>Cause: ' +
                              '<br>'.join([e["cause"] for e in emotions]),
                line=dict(color='#4ECDC4'),
                marker=dict(size=10)
            ))

            fig.update_layout(
                title="Emotional Timeline",
                xaxis_title="Message Sequence",
                yaxis_title="Emotion",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def display_debug_controls(self):
        """Display debug controls"""
        with st.sidebar:
            st.markdown("### Debug Controls")
            st.toggle("Show Debug Info", key="show_debug")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.processing = False
                st.rerun()

    def run(self):
        """Main UI loop with tabs"""
        self.display_header()
        self.display_debug_controls()

        # Create tabs
        tab1, tab2 = st.tabs(["Chat", "Visualization"])

        with tab1:
            # Main chat interface
            chat_container = st.container()
            self.display_chat_history()

            if st.session_state.response_options:
                self.display_response_options(st.session_state.response_options)

            if prompt := st.chat_input("Share your thoughts...", key="chat_input"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt
                })
                st.session_state.processing = True
                st.rerun()

            if st.session_state.processing and not st.session_state.response_options:
                try:
                    last_message = next(msg["content"] for msg in reversed(st.session_state.messages)
                                        if msg["role"] == "user")
                    result = self.model_handler.process_message(last_message)
                    st.session_state.current_emotion = result['emotion_data']
                    self.display_response_options(result['responses'])

                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")
                    st.session_state.processing = False

        with tab2:
            st.markdown("### Emotion Analysis Visualizations")

            if st.session_state.messages and any("emotion_data" in msg for msg in st.session_state.messages):
                # Emotion Processing Map
                st.markdown("#### Current Emotion Processing")
                last_result = {
                    'emotion_data': st.session_state.current_emotion,
                    'graph_insights': self.model_handler.graph_processor.process_emotion(
                        st.session_state.current_emotion['emotion'],
                        st.session_state.current_emotion['cause'],
                        ""
                    )
                } if st.session_state.current_emotion else None

                if last_result:
                    fig = self.visualize_emotion_processing(last_result)
                    st.plotly_chart(fig, use_container_width=True)

                # Emotional Timeline
                st.markdown("#### Emotional Timeline")
                self.display_emotion_timeline()

                # Emotion Statistics
                st.markdown("#### Emotion Statistics")
                emotions = [msg["emotion_data"]["emotion"]
                            for msg in st.session_state.messages
                            if "emotion_data" in msg]

                emotion_counts = Counter(emotions)

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(emotion_counts.keys()),
                        y=list(emotion_counts.values()),
                        marker=dict(color='#4ECDC4')
                    )
                ])

                fig.update_layout(
                    title="Emotion Frequencies",
                    xaxis_title="Emotions",
                    yaxis_title="Frequency",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Start a conversation to see emotional analysis visualizations")


def create_ui(model_handler) -> StreamlitUI:
    """Create and return UI instance"""
    return StreamlitUI(model_handler)