import os
import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# --- Configurations and Initializations ---

# Load environment variables from .env file
load_dotenv()

# 1. Hugging Face Client Setup
hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_api_key:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")

hf_client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    token=hf_api_key
)

# 2. MongoDB Atlas Connection Setup
connection_string = os.getenv("MONGO_CONNECTION_STRING")
if not connection_string:
    raise ValueError("MONGO_CONNECTION_STRING not found in .env file")

# Create a new client and connect to the server
mongo_client = MongoClient(connection_string, server_api=ServerApi('1'))

# Ping the database to test the connection
try:
    mongo_client.admin.command('ping')
    print("✅ Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit() # Exit the script if the database connection fails

# Define your database and collection
db = mongo_client['ideacritic_db']
debates_collection = db['debates']

# --- Core AI Functions ---

def stream_and_collect_response(prompt):
    """
    Calls the HF API, streams the response, and returns the full text.
    """
    full_response = ""
    try:
        response_stream = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True
        )
        
        for token in response_stream:
            # THE FIX IS HERE: We check if token.choices is not empty before accessing it
            if token.choices:
                content = token.choices[0].delta.content
                if content is not None:
                    print(content, end="", flush=True)
                    full_response += content
        
        print("\n")
        return full_response

    except Exception as e:
        error_message = f"An error occurred with the AI model: {e}"
        print(error_message)
        return error_message

def get_agent_response(persona, idea, history, last_statement):
    """
    Generates a response for a specific agent based on the conversation so far.
    """
    if not last_statement:
        prompt = f"You are an AI assistant with the persona of a startup Optimist. Your goal is to analyze the following startup idea: '{idea}'. Provide your initial, positive analysis with 2-3 key strengths."
    else:
        prompt = f"You are an AI assistant with the persona of a startup {persona}. Your goal is to analyze the startup idea: '{idea}'. The conversation history is:\n---\n{history}\n---\nYour task is to respond DIRECTLY to the last statement: '{last_statement}'. If you are the Optimist, rebut the Critic. If you are the Critic, find flaws in the Optimist's points."
    
    return stream_and_collect_response(prompt)

def get_summary(idea, history):
    """
    Generates a final summary after the debate is complete.
    """
    prompt = f"You are an expert Business Analyst. You have observed a debate between an Optimist and a Critic about the startup idea: '{idea}'. Here is the full transcript:\n---\n{history}\n---\nBased on this debate, provide a neutral and balanced summary with 2-3 of the most critical takeaways for the user."
    
    return stream_and_collect_response(prompt)

# --- Main Application Loop ---

def main():
    """
    The main function to run the IdeaCritic session.
    """
    print("\n🚀 Welcome to IdeaCritic! 🚀")
    
    # 1. Get the startup idea and a title
    startup_idea_title = input("Enter a short title for your startup idea: ")
    startup_idea_desc = input("Please describe your startup idea in detail: ")
    
    # 2. Get the number of rounds
    while True:
        try:
            num_rounds = int(input("How many rounds should the agents debate? (e.g., 3): "))
            if num_rounds > 0:
                break
            else:
                print("Please enter a number greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    conversation_history = ""
    last_response = ""
    
    # 3. Start the automated debate loop
    for i in range(num_rounds):
        round_number = i + 1
        print(f"\n\n--- Round {round_number} ---")

        # Optimist's Turn
        print(f"\n Optimist's Turn:")
        optimist_response = get_agent_response("Optimist", startup_idea_desc, conversation_history, last_response)
        conversation_history += f"\nRound {round_number} - Optimist: {optimist_response}"
        last_response = optimist_response
        
        # Critic's Turn
        print(f"\n Critic's Turn:")
        critic_response = get_agent_response("Critic", startup_idea_desc, conversation_history, last_response)
        conversation_history += f"\nRound {round_number} - Critic: {critic_response}"
        last_response = critic_response

    # 4. Generate the final summary
    print("\n\n--- Final Summary ---")
    final_summary = get_summary(startup_idea_desc, conversation_history)
    
    # 5. NEW: Save the entire debate to the database
    try:
        debate_document = {
            "idea_title": startup_idea_title,
            "idea_description": startup_idea_desc,
            "debate_transcript": conversation_history.strip(),
            "final_summary": final_summary,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }
        
        result = debates_collection.insert_one(debate_document)
        print("\n💾 Debate successfully saved to the database!")
        print(f"📄 Document ID: {result.inserted_id}")

    except Exception as e:
        print(f"\n❌ Failed to save debate to the database: {e}")
    
    print("\nDebate complete. Thank you for using IdeaCritic!")

if __name__ == "__main__":
    main()