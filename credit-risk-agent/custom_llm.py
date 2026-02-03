from typing import Any, List, Optional, Callable, Dict
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
import requests
import os

class CustomDatabricksLLM(CustomLLM):
    """
    Wrapper for Databricks Serving Endpoint using direct HTTP requests.
    Uses the logic that you confirmed works.
    """
    endpoint_name: str
    databricks_host: str
    databricks_token: str
    context_window: int = 4096
    num_output: int = 512

    def __init__(self, endpoint_name, host, token, **kwargs):
        super().__init__(
            endpoint_name=endpoint_name, 
            databricks_host=host, 
            databricks_token=token, 
            **kwargs
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.endpoint_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url = f"{self.databricks_host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json",
        }
        
        # Payload matching your working snippet
        payload = {
            "inputs": [prompt],
            "params": {
                "max_new_tokens": self.num_output,
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.9),
            },
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code != 200:
                print(f"DEBUG: Calling URL: {url}")
                print(f"❌ Databricks Error {response.status_code}: {response.text}")
                return CompletionResponse(text=f"Error: {response.status_code}")
                
            result = response.json()
            # Extract text based on your working snippet structure
            # Adjust key "predictions" if your model output format differs (some use "choices")
            raw_text = result.get("predictions", ["No response"])[0]
            
            if raw_text.startswith(prompt):
                clean_text = raw_text[len(prompt):].strip()
            else:
                # Fallback: Sometimes models use specific separators like "### Response:"
                if "### Response:" in raw_text:
                    clean_text = raw_text.split("### Response:")[-1].strip()
                elif "### Answer" in raw_text:
                     # Splitting by the LAST "Answer:" to avoid splitting instructions
                    clean_text = raw_text.split("### Answer")[-1].strip()
                else:
                    clean_text = raw_text

            return CompletionResponse(text=clean_text)

        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return CompletionResponse(text=f"Connection Error: {str(e)}")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # We don't implement streaming for now to keep it simple
        raise NotImplementedError("Streaming not supported for this custom wrapper")
