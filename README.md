# Medical AI Chatbot 🩺

A comprehensive medical chatbot application that provides health consultations and specialist recommendations using AI-powered analysis. Built with Streamlit, LangChain, and Groq LLM.

## Features

### 🤖 AI-Powered Medical Consultation
- **Intelligent Health Analysis**: Uses Groq's LLaMA 3-70B model for medical query understanding
- **Symptom Assessment**: Analyzes user symptoms and health concerns
- **Conversational Memory**: Maintains chat history for contextual responses
- **Concise Responses**: Provides brief, direct medical advice (2-3 sentences)

### 🏥 Specialist Recommendation System
- **RAG-Based Doctor Search**: Uses LangChain and FAISS for intelligent doctor matching
- **Condition Detection**: Automatically identifies serious medical conditions requiring specialist care
- **Smart Referrals**: Recommends appropriate specialists based on detected conditions
- **Doctor Database**: Comprehensive database with doctor information, specializations, and contact details

### 🖥️ User-Friendly Interface
- **Streamlit Web App**: Clean, responsive web interface
- **Real-time Chat**: Interactive chat interface with message history
- **Specialist Cards**: Professional display of recommended doctors
- **Medical Analysis Display**: Clear presentation of condition analysis and recommendations

## Quick Start

### Prerequisites
- Python 3.8+
- Groq API Key
- Required dependencies (see requirements section)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-ai-chatbot.git
cd medical-ai-chatbot
```

2. **Install dependencies**
```bash
pip install streamlit groq python-dotenv langchain-groq langchain-huggingface langchain-community faiss-cpu sentence-transformers
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Add doctor database**
Place your `healthver2.doctors.json` file in the project root directory.

5. **Run the application**
```bash
streamlit run app.py
```

## Usage

### Web Interface
1. Open your browser and go to `http://localhost:8501`
2. Type your health concerns in the chat input
3. Get AI-powered medical advice and specialist recommendations
4. View recommended doctors if serious conditions are detected

### Command Line Interface
```bash
python app.py
```

### Example Interactions

**General Health Query:**
```
User: "I have a mild headache, what should I do?"
Bot: "For mild headaches, try rest, hydration, and over-the-counter pain relievers. Avoid screens and get adequate sleep."
```

**Serious Condition Detection:**
```
User: "I'm having chest pain and shortness of breath"
Bot: "Chest pain with breathing issues needs immediate attention. I'll recommend specialists."
[Shows cardiologist recommendations with contact details]
```

## Architecture

### Core Components

1. **MedicalChatbot Class**: Main chatbot logic and conversation handling
2. **RAG System**: Retrieval-Augmented Generation for doctor recommendations using:
   - LangChain for document processing
   - FAISS for vector storage and similarity search
   - HuggingFace embeddings for text representation
3. **Medical Analysis Engine**: Groq LLM-powered condition detection and severity assessment
4. **Streamlit Interface**: Web application frontend

### Technical Stack
- **Frontend**: Streamlit
- **LLM**: Groq LLaMA 3-70B-8192
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain
- **Language**: Python 3.8+

## Configuration

### Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Doctor Database Format
The `healthver2.doctors.json` should contain doctor information in this format:
```json
[
  {
    "_id": {"$oid": "doctor_id"},
    "name": "Dr. John Smith",
    "specialization": "Cardiology",
    "category": "Heart Specialist",
    "phone": "+1234567890",
    "email": "doctor@example.com",
    "experience": 15,
    "qualifications": ["MD", "FACC"]
  }
]
```

## Features in Detail

### 🔍 Medical Analysis
- **Aggressive Detection**: Identifies serious conditions requiring immediate attention
- **Condition Categories**: Cancer, cardiovascular, neurological, mental health, chronic diseases
- **Specialty Mapping**: Automatically maps conditions to appropriate medical specialties
- **JSON-Structured Analysis**: Provides structured medical assessment data

### 👨‍⚕️ Specialist Recommendations
- **Intelligent Matching**: Uses semantic search to find relevant specialists
- **Comprehensive Information**: Shows doctor name, specialization, experience, and contact details
- **Top-K Retrieval**: Returns most relevant specialists based on condition analysis
- **Lazy Loading**: Loads doctor database only when needed for optimal performance

### 💬 Chat Management
- **Session-Based History**: Maintains separate chat histories for different sessions
- **Context Awareness**: Uses recent conversation history for better responses
- **Memory Management**: Automatically manages conversation length to prevent memory issues
- **Multi-Interface Support**: Both web and command-line interfaces available

## API Reference

### Main Methods

#### `chat(prompt, session_id="default")`
Main chat function that processes user queries and returns medical advice.

**Parameters:**
- `prompt` (str): User's health-related query
- `session_id` (str): Session identifier for chat history

**Returns:**
```python
{
    "response": "AI response text",
    "analysis": {
        "detected_conditions": ["condition1", "condition2"],
        "is_serious": True/False,
        "recommended_specialty": "specialty name",
        "explanation": "analysis explanation"
    },
    "specialists": [
        {
            "doctor_id": "id",
            "name": "Dr. Name",
            "specialization": "specialty",
            "phone": "phone_number",
            "experience": years
        }
    ]
}
```

#### `clear_chat(session_id="default")`
Clears chat history for specified session.

#### `get_specialists_by_specialty(specialty, limit=5)`
Retrieves specialists by medical specialty.

## Security & Privacy

⚠️ **Important Security Notes:**
- This application is for informational purposes only
- Always consult qualified healthcare professionals for medical advice
- Do not share sensitive personal health information
- Conversations are stored temporarily in session memory
- No persistent data storage of user conversations

## Disclaimer

**Medical Disclaimer**: This chatbot provides general medical information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions. Never disregard professional medical advice or delay seeking it because of information received from this chatbot.

***

**Built with ❤️ for better healthcare accessibility**
