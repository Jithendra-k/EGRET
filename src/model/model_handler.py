import os
from typing import Dict, List, Optional
import yaml
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from dotenv import load_dotenv
from peft import PeftModel
from ..graph.graph_processor import GraphProcessor
from ..preference.response_ranker import ResponseRanker
from ..utils.prompt_manager import PromptManager
from ..utils.conversation_manager import ConversationManager


class ModelHandler:
    """Handles interactions with the fine-tuned LLM and coordinates with graph processor"""

    def __init__(self, config_path: str = "configs/config.yaml", hf_token: Optional[str] = None):
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)

        # Setup logging
        logging.basicConfig(level=self.config['system']['log_level'])
        self.logger = logging.getLogger(__name__)

        # Get HuggingFace token
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        if self.config['model']['use_auth_token'] and not self.hf_token:
            raise ValueError("HuggingFace token is required but not provided.")

        # Set device
        if torch.cuda.is_available() and self.config['model']['device'] == 'cuda':
            self.device = torch.device('cuda')
            self.logger.info("Using CUDA device")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU device")

        # Initialize components
        self.model, self.tokenizer = self._load_model()
        self.graph_processor = GraphProcessor(config_path)
        self.prompt_manager = PromptManager()
        self.conversation_manager = ConversationManager()
        self.response_ranker = ResponseRanker()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace"""
        try:
            base_model_id = "unsloth/llama-3.2-3b-instruct-bnb-4bit"  # Changed to Unsloth base model
            adapter_model_id = self.config['model']['model_id']
            self.logger.info(f"Loading model from {adapter_model_id}")

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )

            model = PeftModel.from_pretrained(
                base_model,
                adapter_model_id,
                token=self.hf_token if self.config['model']['use_auth_token'] else None
            )

            if self.device.type == "cpu":
                model = model.to(self.device)

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def detect_emotion(self, text: str, conversation_history: str = "") -> Dict:
        """Task 1: Emotion Detection"""
        try:
            # Create prompt using prompt manager
            prompt = self.prompt_manager.create_emotion_detection_prompt(
                    text=text,
                    conversation_history=conversation_history,
            )

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config['model']['max_length'],
                    temperature=self.config['model']['temperature'],
                    top_p=self.config['model']['top_p'],
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and parse response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("=======================================================")
            print("response: ",response) ##debugging
            parsed_response = self._parse_emotion_response(response)
            print("=======================================================")
            print("Final Parsed Response for emotion and its cause detection: ", parsed_response)

            if parsed_response['emotion'] == 'unknown':
                raise ValueError("Could not detect emotion from model response")

            return parsed_response

        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return {
                'emotion': 'neutral',  # Default fallback
                'cause': 'unclear',
                'confidence': 0.0
            }

    def generate_responses(self, text: str, emotion_data: Dict, graph_insights: Dict, conversation_history: str = "") -> List[Dict]:
        """Task 2: Response Generation"""
        try:
            # Create prompt using prompt manager
            print("=======================================================")
            print("Generating response with:", graph_insights)
            prompt = self.prompt_manager.create_response_generation_prompt(
                text, emotion_data, graph_insights, conversation_history
            )

            # Tokenize and generate multiple responses
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config['model']['max_length'],
                    temperature=self.config['model']['temperature'],
                    top_p=self.config['model']['top_p'],
                    num_return_sequences=self.config['model']['num_responses'],
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Process outputs with response styles
            responses = []
            for idx, output in enumerate(outputs):
                response_text = self.tokenizer.decode(output, skip_special_tokens=True)
                response_text = self.prompt_manager.format_response(response_text)

                # Get response style from graph insights
                styles = graph_insights.get('response_patterns', [])
                style = styles[idx % len(styles)] if styles else "default"

                responses.append({
                    'text': response_text,
                    'style': style,
                    'score': 1.0
                })
            print("===================================================")
            print(responses)
            return responses

        except Exception as e:
            self.logger.error(f"Error generating responses: {str(e)}")
            return [{'text': 'I apologize, but I am having trouble generating a response.',
                     'style': 'error',
                     'score': 0.0}]

    import re

    def _parse_emotion_response(self, response: str) -> Dict:
        try:
            # Get the model output portion
            actual_response = response.split("Response:")[
                -1].strip().lower() if "Response:" in response else response.lower()

            # Extract emotion and cause
            emotion_match = re.search(r"emotion:\s*([^\n:]+)", actual_response, re.IGNORECASE)
            cause_match = re.search(r"cause:\s*([^\n]+)", actual_response, re.IGNORECASE)

            # Take only the first word for emotion, full phrase for cause
            emotion_full = emotion_match.group(1).strip() if emotion_match else 'neutral'
            emotion = emotion_full.split()[0].strip('[]() ') if emotion_full != 'neutral' else 'neutral'

            cause = cause_match.group(1).strip() if cause_match else 'unclear'
            cause = cause.strip('[]() ')

            self.logger.debug(f"Extracted emotion: {emotion}")
            self.logger.debug(f"Extracted cause: {cause}")

            return {
                'emotion': emotion,
                'cause': cause,
                'confidence': 1.0 if emotion != 'neutral' else 0.0
            }

        except Exception as e:
            self.logger.error(f"Error parsing emotion response: {str(e)}")
            return {
                'emotion': 'neutral',
                'cause': 'unclear',
                'confidence': 0.0
            }

    # def _parse_emotion_response(self, response: str) -> Dict:
    #     try:
    #         # First split the response at "Response:" to get only the model's output
    #         if "Response:" in response:
    #             actual_response = response.split("Response:")[-1].strip().lower()
    #         else:
    #             actual_response = response.lower()
    #
    #         # Now match emotion and cause from the actual response
    #         emotion_match = re.search(r"emotion:\s*([^\n]+)", actual_response, re.IGNORECASE)
    #         cause_match = re.search(r"cause:\s*([^\n]+)", actual_response, re.IGNORECASE)
    #
    #         # Extract emotion and cause
    #         emotion = emotion_match.group(1).strip() if emotion_match else 'neutral'
    #         cause = cause_match.group(1).strip() if cause_match else 'unclear'
    #
    #         # Clean up any remaining brackets
    #         emotion = emotion.replace('[', '').replace(']', '').strip()
    #         cause = cause.replace('[', '').replace(']', '').strip()
    #
    #         self.logger.debug(f"Extracted emotion: {emotion}")
    #         self.logger.debug(f"Extracted cause: {cause}")
    #
    #         return {
    #             'emotion': emotion,
    #             'cause': cause,
    #             'confidence': 1.0 if emotion != 'neutral' else 0.0
    #         }
    #
    #     except Exception as e:
    #         self.logger.error(f"Error parsing emotion response: {str(e)}")
    #         return {
    #             'emotion': 'neutral',
    #             'cause': 'unclear',
    #             'confidence': 0.0
    #         }
    def select_response(self, message: str, emotion_data: Dict, responses: List[Dict], selected_idx: int) -> None:
        selected_response = responses[selected_idx]

        # Save to conversation history
        self.conversation_manager.add_exchange(
            message,
            emotion_data,
            selected_response['text']
        )

        # Save preference
        context = {
            'message': message,
            'emotion_data': emotion_data
        }
        self.response_ranker.save_preference(context, selected_response, responses)

        return selected_response

    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize detected emotion to match base patterns"""
        # Comprehensive emotion mapping
        emotion_mapping = {
            'excited': 'excitement',
            'happy': 'joy',
            'joyful': 'joy',
            'delighted': 'joy',
            'elated': 'joy',
            'cheerful': 'joy',
            'ecstatic': 'joy',
            'giddy': 'joy',
            'thrilled': 'joy',
            'merry': 'joy',
            'gleeful': 'joy',
            'overjoyed': 'joy',
            'euphoric': 'joy',
            'jubilant': 'joy',
            'blissful': 'joy',
            'content': 'contentment',
            'peaceful': 'contentment',
            'serene': 'contentment',
            'tranquil': 'contentment',
            'relaxed': 'contentment',
            'calm': 'contentment',
            'satisfied': 'contentment',
            'fulfilled': 'contentment',
            'comfortable': 'contentment',
            'at_ease': 'contentment',
            'sentimental': 'sentimentality',
            'nostalgic': 'nostalgia',
            'melancholy': 'melancholy',
            'wistful': 'melancholy',
            'reflective': 'melancholy',
            'longing': 'melancholy',
            'sad': 'sadness',
            'sorrowful': 'sadness',
            'gloomy': 'sadness',
            'depressed': 'sadness',
            'dejected': 'sadness',
            'downcast': 'sadness',
            'morose': 'sadness',
            'grief-stricken': 'sadness',
            'heartbroken': 'sadness',
            'devastated': 'sadness',
            'distraught': 'sadness',
            'despondent': 'sadness',
            'anguished': 'sadness',
            'miserable': 'sadness',
            'forlorn': 'sadness',
            'desolate': 'sadness',
            'crestfallen': 'sadness',
            'disheartened': 'sadness',
            'dismal': 'sadness',
            'mournful': 'sadness',
            'despairing': 'sadness',
            'blue': 'sadness',
            'down': 'sadness',
            'angry': 'anger',
            'furious': 'anger',
            'enraged': 'anger',
            'irate': 'anger',
            'livid': 'anger',
            'incensed': 'anger',
            'indignant': 'anger',
            'outraged': 'anger',
            'infuriated': 'anger',
            'seething': 'anger',
            'exasperated': 'anger',
            'irritated': 'anger',
            'annoyed': 'anger',
            'vexed': 'anger',
            'aggravated': 'anger',
            'frustrated': 'anger',
            'displeased': 'anger',
            'mad': 'anger',
            'bitter': 'anger',
            'hostile': 'anger',
            'hateful': 'anger',
            'spiteful': 'anger',
            'envenomed': 'anger',
            'venomous': 'anger',
            'afraid': 'fear',
            'scared': 'fear',
            'terrified': 'fear',
            'petrified': 'fear',
            'horrified': 'fear',
            'panic-stricken': 'fear',
            'frightened': 'fear',
            'apprehensive': 'fear',
            'worried': 'fear',
            'nervous': 'fear',
            'uneasy': 'fear',
            'tense': 'fear',
            'on_edge': 'fear',
            'alarmed': 'fear',
            'dread': 'fear',
            'timid': 'fear',
            'timorous': 'fear',
            'skittish': 'fear',
            'shy': 'fear',
            'fearful': 'fear',
            'spooked': 'fear',
            'jumpy': 'fear',
            'startled': 'surprise',
            'astonished': 'surprise',
            'amazed': 'surprise',
            'shocked': 'surprise',
            'bewildered': 'surprise',
            'stunned': 'surprise',
            'dumbfounded': 'surprise',
            'flabbergasted': 'surprise',
            'astounded': 'surprise',
            'awestruck': 'surprise',
            'wide-eyed': 'surprise',
            'surprised': 'surprise',
            'disgusted': 'disgust',
            'revolted': 'disgust',
            'sickened': 'disgust',
            'nauseated': 'disgust',
            'repulsed': 'disgust',
            'queasy': 'disgust',
            'appalled': 'disgust',
            'abhorrent': 'disgust',
            'averse': 'disgust',
            'proud': 'pride',
            'accomplished': 'pride',
            'unflinching': 'confidence',
            'poised': 'confidence',
            'determined': 'determination',
            'resolute': 'determination',
            'unwavering': 'determination',
            'steadfast': 'determination',
            'persevering': 'determination',
            'dedicated': 'determination',
            'caring': 'care',
            'compassionate': 'care',
            'nurturing': 'care',
            'empathetic': 'care',
            'sympathetic': 'care',
            'concerned': 'care',
            'protective': 'care',
            'supportive': 'care',
            'loving': 'love',
            'affectionate': 'love',
            'adoring': 'love',
            'infatuated': 'love',
            'smitten': 'love',
            'enamored': 'love',
            'besotted': 'love',
            'devoted': 'love',
            'faithful': 'trust',
            'trusting': 'trust',
            'assured': 'trust',
            'confident': 'trust',
            'secure': 'trust',
            'certain': 'trust',
            'self-assured': 'trust',
            'hopeful': 'hope',
            'optimistic': 'hope',
            'encouraged': 'hope',
            'motivated': 'hope',
            'inspired': 'hope',
            'eager': 'anticipation',
            'anticipating': 'anticipation',
            'expectant': 'anticipation',
            'impatient': 'anticipation',
            'anxious': 'anticipation',
            'prepared': 'preparedness',
            'ready': 'preparedness',
            'organized': 'preparedness',
            'composed': 'preparedness',
            'focused': 'preparedness',
            'ashamed': 'shame',
            'guilty': 'guilt',
            'embarrassed': 'embarrassment',
            'self-conscious': 'embarrassment',
            'uncomfortable': 'embarrassment',
            'mortified': 'embarrassment',
            'regretful': 'regret',
            'remorseful': 'regret',
            'responsible': 'responsibility',
            'lonely': 'loneliness',
            'isolated': 'loneliness',
            'disconnected': 'loneliness',
            'yearning': 'loneliness',
            'jealous': 'jealousy',
            'envious': 'jealousy',
            'resentful': 'jealousy',
            'competitive': 'jealousy',
            'insecure': 'insecurity'
        }

        # Clean the emotion string
        emotion = emotion.lower().strip()

        # Check direct match with base patterns first
        if emotion in self.graph_processor.emotion_graph.nodes():
            return emotion

        # Try mapped emotion
        mapped_emotion = emotion_mapping.get(emotion)
        if mapped_emotion and mapped_emotion in self.graph_processor.emotion_graph.nodes():
            return mapped_emotion

        # Default fallback
        return 'neutral'

    @torch.no_grad()
    def process_message(self, text: str) -> Dict:
        """Main message processing pipeline"""
        try:
            # Get the conversation history if exists
            conversation_history = self.conversation_manager.get_formatted_history()
            # Step 1: Emotion Detection
            emotion_data = self.detect_emotion(text, conversation_history)

            # Step 2: Get graph insights
            graph_insights = self.graph_processor.process_emotion(
                emotion=emotion_data['emotion'],
                cause=emotion_data['cause'],
                context=text
            )

            # Step 3: Generate responses
            responses = self.generate_responses(
                text,
                emotion_data,
                graph_insights,
                conversation_history
            )

            # # Save conversation
            # if responses:
            #     self.conversation_manager.add_exchange(text, emotion_data, responses[0]['text'])

            # Save graph state periodically
            if self.graph_processor.should_save():
                self.graph_processor.save_graph()

            return {
                'emotion_data': emotion_data,
                'graph_insights': graph_insights,
                'responses': responses
            }

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise