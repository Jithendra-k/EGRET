EGRET/
├── .gitignore
├── requirements.txt         # Include streamlit and other dependencies
├── README.md
├── app.py                  # Main Streamlit entry point
│
├── configs/
│   ├── config.yaml         # Main configuration file
│   └── graph_config.yaml   # Graph-specific configurations
│
├── data/
│   ├── emotion_patterns/
│   │   └── base_patterns.json    # Initial emotion relationships
│   └── preference_history/
│       └── preferences.json      # User interaction history
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── model_handler.py      # Your HuggingFace model logic
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graph_processor.py    # Graph management
│   │
│   ├── preference/
│   │   ├── __init__.py
│   │   └── response_ranker.py    # Response ranking
│   │
│   ├── ui/                       # New UI directory
│   │   ├── __init__.py
│   │   ├── pages/               # Multiple pages if needed
│   │   │   ├── __init__.py
│   │   │   ├── chat.py         # Main chat interface
│   │   │   └── analysis.py     # Optional analysis/insights page
│   │   ├── components/         # Reusable UI components
│   │   │   ├── __init__.py
│   │   │   ├── chat_box.py    # Chat interface component
│   │   │   └── response_selector.py  # Response selection component
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── ui_helpers.py   # UI utility functions
│   │
│   └── utils/
│       ├── __init__.py
│       └── prompt_manager.py    # Prompt templates
│
├── notebooks/
│   └── experiments.ipynb        # For testing and demonstrations
│
└── tests/
    ├── __init__.py
    ├── test_graph.py
    └── test_ranking.py