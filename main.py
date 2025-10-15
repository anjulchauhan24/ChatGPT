from flask import Flask, render_template, jsonify, request, stream_with_context, Response
from flask_cors import CORS
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store chat sessions
chat_sessions = {}

# Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"  # or "gpt-4" if you have access
SYSTEM_MESSAGE = "You are ChatGPT, a helpful AI assistant created by OpenAI. You provide accurate, helpful, and friendly responses."

@app.route("/")
def home():
    """Serve the main HTML page"""
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint with OpenAI integration
    Supports both streaming and non-streaming responses
    """
    try:
        data = request.json
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        stream = data.get("stream", False)
        model = data.get("model", DEFAULT_MODEL)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({
                "error": "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file"
            }), 500
        
        # Initialize session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Add user message to session
        chat_sessions[session_id].append({
            "role": "user",
            "content": question,
            "timestamp": time.time()
        })
        
        # Generate response with OpenAI
        if stream:
            return Response(
                stream_with_context(generate_stream_response(question, session_id, model)),
                mimetype='text/event-stream'
            )
        else:
            response_text = generate_openai_response(question, session_id, model)
            
            # Add assistant message to session
            chat_sessions[session_id].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": time.time()
            })
            
            return jsonify({
                "success": True,
                "result": response_text,
                "session_id": session_id,
                "model": model
            })
    
    except Exception as e:
        error_message = str(e)
        print(f"Error in chat endpoint: {error_message}")
        
        # Handle specific OpenAI errors
        if "api_key" in error_message.lower():
            return jsonify({
                "error": "Invalid OpenAI API key. Please check your .env file"
            }), 401
        elif "quota" in error_message.lower():
            return jsonify({
                "error": "OpenAI quota exceeded. Please check your billing"
            }), 429
        else:
            return jsonify({"error": f"An error occurred: {error_message}"}), 500

def generate_openai_response(question, session_id, model):
    """
    Generate a response using OpenAI API (non-streaming)
    """
    try:
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]
        
        # Add conversation history (last 20 messages to stay within token limits)
        history = chat_sessions[session_id][-20:]
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        return response_text
    
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        raise

def generate_stream_response(question, session_id, model):
    """
    Generate a streaming response using OpenAI API
    """
    try:
        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]
        
        # Add conversation history
        history = chat_sessions[session_id][-20:]
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call OpenAI API with streaming
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        full_response = ""
        
        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
        
        # Save complete response to session
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": full_response,
            "timestamp": time.time()
        })
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """Get all chat sessions"""
    sessions = []
    for session_id, messages in chat_sessions.items():
        if messages:
            # Find first user message for title
            first_user_msg = next((m for m in messages if m["role"] == "user"), None)
            sessions.append({
                "id": session_id,
                "title": first_user_msg["content"][:50] if first_user_msg else "New Chat",
                "message_count": len(messages),
                "last_updated": messages[-1]["timestamp"]
            })
    
    # Sort by last updated (most recent first)
    sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    return jsonify({"sessions": sessions})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get specific chat session"""
    if session_id in chat_sessions:
        return jsonify({
            "session_id": session_id,
            "messages": chat_sessions[session_id]
        })
    return jsonify({"error": "Session not found"}), 404

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({"success": True, "message": "Session deleted"})
    return jsonify({"error": "Session not found"}), 404

@app.route("/api/clear", methods=["POST"])
def clear_all():
    """Clear all sessions"""
    chat_sessions.clear()
    return jsonify({"success": True, "message": "All sessions cleared"})

@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available OpenAI models"""
    models = [
        {
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "description": "Fast and efficient for most tasks"
        },
        {
            "id": "gpt-4",
            "name": "GPT-4",
            "description": "Most capable model, better reasoning"
        },
        {
            "id": "gpt-4-turbo-preview",
            "name": "GPT-4 Turbo",
            "description": "Faster GPT-4 with 128K context"
        }
    ]
    return jsonify({"models": models})

@app.route("/health")
def health():
    """Health check endpoint"""
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    
    return jsonify({
        "status": "healthy",
        "sessions": len(chat_sessions),
        "openai_configured": api_key_set,
        "default_model": DEFAULT_MODEL
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ChatGPT Clone with OpenAI Integration")
    print("=" * 70)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API Key: Configured (ends with ...{api_key[-4:]})")
    else:
        print("❌ OpenAI API Key: NOT CONFIGURED")
        print("⚠️  Please set OPENAI_API_KEY in your .env file")
        print("   Get your API key from: https://platform.openai.com/api-keys")
    
    print(f"\n📝 Available Endpoints:")
    print(f"   • GET    /                     - Web interface")
    print(f"   • POST   /api/chat             - Send message (OpenAI)")
    print(f"   • GET    /api/sessions         - Get all sessions")
    print(f"   • GET    /api/sessions/<id>    - Get session")
    print(f"   • DELETE /api/sessions/<id>    - Delete session")
    print(f"   • POST   /api/clear            - Clear all sessions")
    print(f"   • GET    /api/models           - Get available models")
    print(f"   • GET    /health               - Health check")
    
    print(f"\n🤖 Default Model: {DEFAULT_MODEL}")
    print("=" * 70)
    print(f"🌐 Server running at: http://localhost:{os.getenv('PORT', 5000)}")
    print("=" * 70)
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 5000))
    )