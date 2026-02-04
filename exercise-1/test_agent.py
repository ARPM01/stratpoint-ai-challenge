"""
Test script to verify the solar prediction agent functionality
"""

import os

from langchain_core.messages import HumanMessage
from model_utils import resources
from solar_prediction_agent import SolarPredictionAgent

# Set working directory to exercise-1
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load resources
print("Loading models and data...")
resources.load(base_path=os.path.dirname(os.path.abspath(__file__)))

# Initialize and build agent
print("\nBuilding agent...")
agent_instance = SolarPredictionAgent()
agent = agent_instance.build()

# Test queries
test_queries = [
    "What is the solar output if the temperature is 30 degrees and it's sunny in Albury?",
    "What's the solar output for Sydney today?",
    "How much solar power can I generate in Canberra during summer?",
    "Which has better solar potential today: Darwin or Hobart?",
    "I'm installing solar panels in Newcastle - what can I expect on a typical February day?",
]

print("\n" + "=" * 80)
print("Testing Solar Prediction Agent")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Test Query {i}: {query}")
    print(f"{'='*80}\n")

    messages = [HumanMessage(content=query)]

    try:
        result = agent.invoke({"messages": messages})

        # Display the final response
        if "messages" in result and result["messages"]:
            for message in result["messages"]:
                if hasattr(message, "content") and message.content:
                    print(f"{message.type}: {message.content}\n")
        else:
            print("\nNo response generated.\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
