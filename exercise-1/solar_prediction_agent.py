"""
Solar Prediction Agent using LangChain's create_agent.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from solar_tools import (
    get_seasonal_weather_defaults,
    lookup_location,
    predict_solar_output,
)

# Load environment variables
load_dotenv()


class SolarPredictionAgent:
    """
    Agent for predicting solar PV output based on weather conditions.
    Uses LangChain's create_agent with Ollama LLM.
    """

    def __init__(self, model_name=None, base_url=None):
        """
        Initialize the Solar Prediction Agent.

        Args:
            model_name: Name of the Ollama model to use. Defaults to OLLAMA_LLM from .env or "qwen2.5:7b".
            base_url: Base URL for Ollama API. Defaults to OLLAMA_BASE_URL from .env.
        """
        self.model_name = model_name or os.getenv("OLLAMA_LLM", "qwen2.5:7b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self.llm = None
        self.agent = None

    def build(self):
        """
        Builds the LangChain agent with the Solar Prediction tool.

        Returns:
            The created agent executor.
        """
        self.llm = ChatOllama(
            model=self.model_name, temperature=0, base_url=self.base_url
        )
        current_date = datetime.now().strftime("%B %d, %Y")

        # Create system prompt that instructs the agent to ask for location and month
        system_message = f"""You are a Solar Prediction Assistant for locations in Australia.

CURRENT DATE: {current_date}
CURRENT MONTH: {datetime.now().month}
CURRENT SEASON IN AUSTRALIA: {'Summer (Dec-Feb)' if datetime.now().month in [12, 1, 2] else 'Autumn (Mar-May)' if datetime.now().month in [3, 4, 5] else 'Winter (Jun-Aug)' if datetime.now().month in [6, 7, 8] else 'Spring (Sep-Nov)'}

IMPORTANT WORKFLOW:
1. If location is not provided, ask the user which city/location in Australia they want predictions for.
2. Use the lookup_location tool to validate and get coordinates for the location.
3. If month is not provided, ask the user which month they want the prediction for (or assume current month).
4. Use the get_seasonal_weather_defaults tool with the month to get typical weather conditions for that season.
5. Present the seasonal defaults to the user and ask if they want to:
   a) Use these default values for the prediction
   b) Provide their own specific weather parameters
   c) Modify some of the defaults
6. Finally, use the predict_solar_output tool with either the defaults or user-provided values.

Notes:
- The get_seasonal_weather_defaults tool provides realistic weather parameters based on Australian seasons
- If the user provides specific weather parameters, use those instead of defaults
- If the user provides a location and/or month in their initial query, use them directly
- Be helpful and guide users through the process
"""

        self.agent = create_agent(
            self.llm,
            tools=[
                lookup_location,
                get_seasonal_weather_defaults,
                predict_solar_output,
            ],
            system_prompt=system_message,
        )
        return self.agent

    def get_agent(self):
        """
        Get the agent, building it if necessary.

        Returns:
            The agent executor.
        """
        if self.agent is None:
            self.build()
        return self.agent


def build_agent(model_name=None, base_url=None):
    """
    Convenience function to build and return a Solar Prediction Agent.
    Maintains backward compatibility with existing code.

    Args:
        model_name: Name of the Ollama model to use.
        base_url: Base URL for Ollama API.

    Returns:
        The created agent executor.
    """
    agent_instance = SolarPredictionAgent(model_name=model_name, base_url=base_url)
    return agent_instance.build()
