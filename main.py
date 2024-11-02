import os
from src.model.model_handler import ModelHandler
from dotenv import load_dotenv
import logging


def test_inference():
    # Initialize model handler
    try:
        print("Initializing model handler...")
        model_handler = ModelHandler()

        test_messages = [
            "I just got promoted at work! I'm so excited!",
            "I failed my driving test today. Feeling really disappointed.",
            "My best friend surprised me with birthday gifts!"
        ]

        print("\n=== Testing EGRET Inference ===\n")

        for message in test_messages:
            print(f"\nProcessing message: {message}")
            print("-" * 50)

            try:
                result = model_handler.process_message(message)

                print("\nEmotion Analysis:")
                print(f"Emotion: {result['emotion_data']['emotion']}")
                print(f"Cause: {result['emotion_data']['cause']}")

                if 'graph_insights' in result:
                    print("\nGraph Insights:")
                    print(
                        f"Related Emotions: {', '.join([e['emotion'] for e in result['graph_insights']['related_emotions']])}")

                if 'responses' in result:
                    print("\nGenerated Responses:")
                    for idx, response in enumerate(result['responses'], 1):
                        print(f"\n{idx}. Style: {response['style']}")
                        print(f"   Response: {response['text']}")

            except Exception as e:
                print(f"Error processing message: {str(e)}")

            print("\n" + "=" * 50)

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load environment variables
    load_dotenv()

    # Check for HuggingFace token
    if not os.getenv('HUGGINGFACE_TOKEN'):
        print("Please set your HUGGINGFACE_TOKEN in .env file")
        exit(1)

    # Run test
    test_inference()