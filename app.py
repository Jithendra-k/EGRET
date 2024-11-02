import streamlit as st
from src.model.model_handler import ModelHandler
from src.ui.streamlit_ui import create_ui


def main():
    # Initialize model handler
    @st.cache_resource
    def load_model_handler():
        return ModelHandler()

    # Create UI
    model_handler = load_model_handler()
    ui = create_ui(model_handler)

    # Run UI
    ui.run()


if __name__ == "__main__":
    main()