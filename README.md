# EGRET: Emotion Graph-Enhanced Response Generation with Transformative preference learning

EGRET is an empathetic conversational AI system that uses graph-based emotion understanding and preference learning to generate context-aware, emotionally appropriate responses.

## Project Overview

EGRET combines:
- Emotion and cause detection
- Graph-based emotional context tracking
- Response generation with preference optimization
- All integrated into a single fine-tuned LLaMA model

## Project Structure
```
EGRET/
├── .env                    # Environment variables
├── .gitignore
├── requirements.txt        # Project dependencies
├── app.py                  # Main Streamlit application
├── main.py                 # Phase 1 Implementation
│
├── configs/
│   ├── config.yaml         # Main configuration
│   └── graph_config.yaml   # Graph-specific configurations
│
├── data/
│   ├── emotion_patterns/
│   │   ├── base_patterns.json    # 32 EmpatheticDialogues emotions
│   │   └── current_graph.json    # [Generated] Current graph state
│   └── preference_history/
│       └── preferences.json      # [Generated] User interaction history
│
├── src/
│   ├── model/
│   │   └── model_handler.py      # LLM interaction handler
│   │
│   ├── graph/
│   │   └── graph_processor.py    # Emotion graph management
│   │
│   ├── preference/
│   │   └── response_ranker.py    # Response ranking system
│   │
│   ├── ui/
│   │   └── streamlit_ui.py       # Streamlit interface
│   │
│   └── utils/
│   │   ├── conversation_manager.py    # Conversation History Management
│       └── prompt_manager.py     # Prompt management
│
└── tests/
    └── test_model.py            # Model testing
```

## Installation

1. Create a Python virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file with your HuggingFace token:
```
HUGGINGFACE_TOKEN=your_token_here
```

## Configuration

1. Update configs/config.yaml:
```yaml
model:
  model_id: "Jithendra-k/EGRET_LLM"
  use_auth_token: true
  device: "cpu"  # or "cuda" if available
  max_length: 512
  temperature: 0.7
  top_p: 0.9
  num_responses: 3

system:
  log_level: "INFO"
  debug: false
```

## Testing

1. Test basic model functionality:
```bash
python tests/test_model.py
```

## Running the Application

1. Start the Streamlit interface:
```bash
streamlit run app.py
```

## Features

- 🎯 Emotion Detection: Identifies emotions and their causes from user input
- 🕸️ Emotion Graph: Tracks emotional context and relationships
- 💬 Response Generation: Creates empathetic, context-aware responses
- 📊 Preference Learning: Improves responses based on user feedback
- 🎨 Interactive UI: User-friendly Streamlit interface

## Dependencies

Core requirements:
- streamlit>=1.24.0
- torch>=2.0.0
- transformers>=4.31.0
- networkx>=3.1
- plotly>=5.15.0
- python-dotenv>=1.0.0
- pyyaml>=6.0.0

## Development Status

Current implementation status:
- ✅ Model Handler: Complete
- ✅ Graph Processor: Complete
- ✅ Base Emotion Patterns: Complete
- ✅ UI Components: Complete
- ✅ Preference Optimization: Complete

## Usage Example

```python
from src.model.model_handler import ModelHandler

# Initialize model
model = ModelHandler()

# Process a message
result = model.process_message(
    "I just got promoted at work! I'm so excited!"
)

# Print results
print(f"Emotion: {result['emotion_data']['emotion']}")
print(f"Response: {result['responses'][0]['text']}")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
