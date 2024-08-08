# from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    ModerationHAP,
    ModerationHAPInput,
    ModerationHAPOutput,
    ModerationParameters,
    SystemMessage,
    TextGenerationParameters,
    TextGenerationReturnOptions
)

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
# load_dotenv()


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


parameters = TextGenerationParameters(
    decoding_method=DecodingMethod.SAMPLE, max_new_tokens=128, min_new_tokens=30, temperature=0.7, top_k=50, top_p=1
)

client = Client(credentials=Credentials.from_env())
model_id = "meta-llama/llama-3-70b-instruct"

def prompt_model(prompt):
    responses = []
    for response in client.text.generation.create(
            model_id=model_id,
            inputs=prompt,
            parameters=TextGenerationParameters(
                max_new_tokens=1024,
                min_new_tokens=32,
                return_options=TextGenerationReturnOptions(
                    input_text=True,
                    generated_tokens=True,
                    token_logprobs=True
                ),
                decoding_method='greedy',
            ),
        ):
            responses.append(response.results[0].generated_text)
    return responses

prompts = ["How can I start?", "How can I start?", "How can I start?"]
print("".join(prompt_model(prompts)))

# print(heading("Continue with a conversation"))
# prompt = "How can I start?"
# response = client.text.chat.create(
#     messages=[HumanMessage(content=prompt)],
#     moderations=ModerationParameters(
#         hap=ModerationHAP(
#             input=ModerationHAPInput(enabled=True, threshold=0.8),
#             output=ModerationHAPOutput(enabled=True, threshold=0.8),
#         )
#     ),
#     conversation_id=conversation_id,
#     use_conversation_parameters=True,
# )
# print(f"Request: {prompt}")
# print(f"Response: {response.results[0].generated_text}")