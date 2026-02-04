"""
Gradio UI for Solar Prediction Agent
"""

import os

import gradio as gr
from langchain_core.messages import HumanMessage
from model_utils import resources
from solar_prediction_agent import SolarPredictionAgent

os.chdir(os.path.dirname(os.path.abspath(__file__)))

resources.load(base_path=os.path.dirname(os.path.abspath(__file__)))

agent_instance = SolarPredictionAgent()
agent = agent_instance.build()

# Global message history
message_history = []


def predict_solar(message, history):
    """Process user query and return agent response with full conversation history"""
    # Add new user message to history
    message_history.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": message_history})

        # Format all messages to show the full conversation
        formatted_messages = []

        if "messages" in result and result["messages"]:
            # Update message history with agent's response
            message_history.clear()
            message_history.extend(result["messages"])

            for msg in result["messages"][1:]:  # Skip the initial user message
                msg_type = type(msg).__name__

                if msg_type == "AIMessage":
                    # Check if it has tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_info = []
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            tool_info.append(f"**Tool Call**: {tool_name}")
                        formatted_messages.append("\n".join(tool_info))

                    # Add AI response content if present
                    if hasattr(msg, "content") and msg.content:
                        formatted_messages.append(f"**AI**: {msg.content}")

                elif msg_type == "ToolMessage":
                    tool_name = getattr(msg, "name", "unknown")
                    content = getattr(msg, "content", "")
                    formatted_messages.append(
                        f"**Tool ({tool_name})**: {content[:200]}{'...' if len(content) > 200 else ''}"
                    )

                elif msg_type == "HumanMessage":
                    if hasattr(msg, "content") and msg.content:
                        formatted_messages.append(f"**User**: {msg.content}")

        if formatted_messages:
            return "\n\n".join(formatted_messages)

        return "No response generated."

    except Exception as e:
        return f"Error: {str(e)}"


# Sample queries
examples = [
    "What is the solar output if the temperature is 30 degrees and it's sunny in Albury?",
    "What's the solar output for Sydney today?",
    "How much solar power can I generate in Canberra during summer?",
    "Which has better solar potential today: Darwin or Hobart?",
    "I'm installing solar panels in Newcastle - what can I expect on a typical February day?",
]


with gr.Blocks(title="Solar Prediction Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
        # Solar PV Output Prediction Agent
        
        Ask questions about solar panel output predictions based on weather conditions.
        The agent uses machine learning models to estimate daily solar PV output (kWh/kWp).
        """)

    chatbot = gr.Chatbot(label="Chat with Solar Prediction Agent", height=400)

    msg = gr.Textbox(
        label="Your Question",
        placeholder="Ask about solar output predictions...",
        lines=2,
    )

    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")

    gr.Examples(examples=examples, inputs=msg, label="Sample Queries")

    def process_query(user_message, chat_history):
        """Process query and maintain conversation history"""
        if not user_message:
            return chat_history, ""

        # Get bot response with full conversation
        bot_message = predict_solar(user_message, chat_history)

        # Append to existing conversation
        updated_history = chat_history + [
            gr.ChatMessage(role="user", content=user_message),
            gr.ChatMessage(role="assistant", content=bot_message),
        ]

        return updated_history, ""

    # Event handlers
    msg.submit(process_query, [msg, chatbot], [chatbot, msg], queue=False)
    submit.click(process_query, [msg, chatbot], [chatbot, msg], queue=False)

    def clear_conversation():
        """Clear both UI and message history"""
        global message_history
        message_history = []
        return []

    clear.click(clear_conversation, None, chatbot, queue=False)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting Solar Prediction Agent UI...")
    print("=" * 80 + "\n")
    demo.launch()
