import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModelTester:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Get HuggingFace token
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            raise ValueError("Please set HUGGINGFACE_TOKEN in .env file")

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        logger.info(f"Loading model: {self.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device.type == "cpu":
            model = model.to(self.device)

        return model, tokenizer

    def detect_emotion(self, text: str) -> dict:
        """Test emotion detection"""
        prompt = f"""Given the text below, identify the emotion being expressed and its cause.
Provide your answer in the following format:
emotion: [identified emotion]
cause: [cause of the emotion]

Text: {text}

Response:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Raw emotion detection response:\n{response}")

            return self._parse_emotion_response(response)

        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return None

    def generate_response(self, text: str, emotion: str, cause: str) -> str:
        """Test response generation"""
        prompt = f"""Generate an empathetic response for someone who is feeling {emotion} because {cause}.
The response should be understanding and supportive. dont give any extra explanation. just give the response to that text.

Original message: {text}

Response:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Raw response generation:\n{response}")

            # Extract the response part
            if "Response:" in response:
                return response.split("Response:")[-1].strip()
            return response.strip()

        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return None

    def _parse_emotion_response(self, response: str) -> dict:
        """Parse the emotion detection response"""
        try:
            # Get the part after 'Response:'
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()

            lines = [line.strip() for line in response.split('\n') if line.strip()]
            result = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    result[key] = value

            return result

        except Exception as e:
            logger.error(f"Error parsing emotion response: {str(e)}")
            return None


def main():
    # Test messages with different emotions
    test_messages = [
        "I just got promoted at work! I can't believe it!",
    ]

    try:
        # Initialize tester
        tester = SimpleModelTester()

        print("\n=== Testing Model Functionality ===\n")

        for msg in test_messages:
            print(f"\nTesting message: {msg}")
            print("-" * 50)

            # Test emotion detection
            print("\nTesting Emotion Detection:")
            emotion_result = tester.detect_emotion(msg)
            if emotion_result:
                print(f"Detected Emotion: {emotion_result.get('emotion', 'unknown')}")
                print(f"Detected Cause: {emotion_result.get('cause', 'unknown')}")

                # Test response generation
                print("\nTesting Response Generation:")
                response = tester.generate_response(
                    msg,
                    emotion_result.get('emotion', ''),
                    emotion_result.get('cause', '')
                )
                if response:
                    print(f"Generated Response: {response}")

            print("\n" + "=" * 50)

    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")


if __name__ == "__main__":
    main()