from llama_cpp import Llama
from typing import Dict
from genson import SchemaBuilder
import requests
import json


class LLMClassifier:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.builder = SchemaBuilder(schema_uri=None)
        self.builder.add_object({"type": "object", "properties": {}})
        self.llm = Llama(
            model_path=self.config.get("model_path", ""),
            chat_format="chatml",
            verbose=self.config.get("verbose", False),
            seed=0,
        )

    def _generate_schema(self) -> Dict:
        """
        Generates a JSON schema based on the config structure
        """

        struct = self.config.get("struct", {})
        for key in struct:
            if self.config.get(key):
                self.builder.add_object({key: struct[key]})

        return self.builder.to_schema()

    def _generate_prompt(self, content: str) -> str:
        """
        Generates a prompt for the LLM to return a response in JSON format give certain conditions.
        """

        base_prompt = f"Given the content perform an IAB and demographics classification based on your knowledge. \
                        Webpage to classify is as follows: {content}."

        prompts = {
            "IAB_categories": "Classify the content into IAB categories version 2. Returns minimum 4 categories \
                               that are related to the content.",
            "Age_groups": "Also, choose age groups from the following groups: teen, young adults, middle aged, boomers, \
                            which would be interested in the above content. Returns at least 1 group.",
            "Gender": "Also, choose gender from the following groups: Male, Female, Other, which would be interested \
                      in the above content. Returns at least 1 group.",
            "Income": "Also, choose income groups from the following groups: Low income, Medium income, High income, \
                        which would be interested in the above content. Returns at least 1 group.",
            "Topics": "Also, return minimum 3 general topics that are related to the content.",
        }

        for key, prompt_str in prompts.items():
            if self.config.get(key):
                base_prompt += f" {prompt_str}"

        prompt = f"{base_prompt} Return response in complete json format."

        return prompt

    def get_response(self, content: str) -> Dict:
        """
        Returns a classification based on LLM
        """

        schema = self._generate_schema()
        prompt = self._generate_prompt(content=content)

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that classifies and generates output in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_object",
                "schema": schema,
            },
            max_tokens=self.config.get("max_tokens", 150),
            temperature=self.config.get("temperature", 0.1),
        )
        response = response["choices"][0]["message"]["content"]
        response = json.loads(response)
        return response


class APIClassifier(LLMClassifier):
    def __init__(self, config: Dict) -> None:
        self.llm = None
        self.host = config.get("host", "http://localhost:8000")
        super().__init__(config)

    def get_response(self, content: str) -> Dict:
        """
        Sends request to LLM API-server
        """

        schema = self._generate_schema()
        prompt = self._generate_prompt(content=content)
        endpoint = f"{self.host}/v1/chat/completions"
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that classifies and generates output in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object", "schema": schema},
            "max_tokens": self.config.get("max_tokens", 150),
            "temperature": self.config.get("temperature", 0.1),
        }

        response = requests.post(endpoint, json=data, headers=headers)

        if response.status_code == 200:
            response = response.json()
            response = response["choices"][0]["message"]["content"]
            return response
        else:
            return {"message": "Unknown error!"}
