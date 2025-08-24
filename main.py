import groq
import os
import json
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import streamlit as st

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document

class MedicalChatbot:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = self.groq_api_key
        self.groq_client = None
        self.vectorstore = None
        self.rag_chain = None
        self.doctors_data = []
        self.chat_histories = {}
        self._initialize()
    
    def _initialize(self):
        try:
            self.groq_client = groq.Client(api_key=self.groq_api_key)
            print("Medical AI Chatbot initialized successfully!")
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
    
    def setup_doctors_rag_system(self):
        try:
            with open('healthver2.doctors.json', 'r') as file:
                doctors_data = json.load(file)
            documents = []
            for doctor in doctors_data:
                doc_text = f"Doctor Name: {doctor.get('name', '')}.\n"
                doc_text += f"Specialization: {doctor.get('specialization', '')}.\n"
                doc_text += f"Category: {doctor.get('category', '')}.\n"
                doc_text += f"Experience: {doctor.get('experience', '')} years.\n"
                doc_text += f"Phone: {doctor.get('phone', '')}.\n"
                doc_text += f"Email: {doctor.get('email', '')}.\n"
                doc_text += f"Qualifications: {', '.join(doctor.get('qualifications', []))}.\n"
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "doctor_id": doctor.get("_id", {}).get("$oid", ""),
                        "name": doctor.get("name", ""),
                        "specialization": doctor.get("specialization", ""),
                        "category": doctor.get("category", ""),
                        "phone": doctor.get("phone", ""),
                        "experience": doctor.get("experience", 0)
                    }
                )
                documents.append(doc)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            return doctors_data, vectorstore
        except Exception as e:
            print(f"Error setting up RAG system: {e}")
            return [], None
    
    def get_normal_chatbot_response(self, query, chat_history):
        history_text = ""
        for msg in chat_history[-4:]:
            if hasattr(msg, 'content'):
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"
        system_prompt = """
You are a medical AI. Give brief, direct answers. No long explanations.

Rules:
- 2-3 sentences max
- Direct and factual
- Skip pleasantries
- If serious condition mentioned, acknowledge briefly then say "I'll recommend specialists"
- For general questions, give quick practical advice

Conversation History:
{history}

Be concise and helpful.
"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt.format(history=history_text)},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting chatbot response: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def analyze_medical_query(self, query):
        system_prompt = """
Analyze this query for medical conditions requiring specialist care.

Mark as serious if ANY of these are mentioned:
- Cancer, tumors, masses, lumps
- Heart/chest pain, cardiovascular issues
- Neurological symptoms (seizures, stroke, severe headaches)
- Mental health crises, depression, anxiety disorders
- Diabetes complications
- Kidney/liver problems
- Breathing difficulties
- Chronic pain
- Surgical needs
- Any specific disease names

Respond ONLY with JSON:
{
  "detected_conditions": ["condition1"],
  "is_serious": true/false,
  "recommended_specialty": "specialty name",
  "explanation": "one sentence why"
}

Be aggressive - when in doubt, mark as serious.
"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            print(f"Error analyzing query: {e}")
            return {"detected_conditions": [], "is_serious": False, "recommended_specialty": "", "explanation": ""}
    
    def find_specialists_rag(self, specialty, top_k=3):
        query = f"Find doctors specializing in {specialty}"
        docs = self.vectorstore.similarity_search(query, k=top_k)
        relevant_doctors = []
        for doc in docs:
            relevant_doctors.append(doc.metadata)
        return relevant_doctors, specialty
    
    def create_rag_chain(self, vectorstore, llm):
        system_prompt = (
            "Find relevant doctors based on the medical specialty needed. "
            "Be brief and direct.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
        return rag_chain
    
    def lazy_load_rag(self):
        if self.vectorstore is None:
            print("Loading specialist database...")
            doctors_data_loaded, vectorstore_loaded = self.setup_doctors_rag_system()
            self.doctors_data = doctors_data_loaded
            self.vectorstore = vectorstore_loaded
            if self.vectorstore is not None:
                llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=self.groq_api_key)
                self.rag_chain = self.create_rag_chain(self.vectorstore, llm)
    
    def chat(self, prompt: str, session_id: str = "default") -> Dict[str, Any]:
        try:
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = []
            chat_history = self.chat_histories[session_id]
            analysis = self.analyze_medical_query(prompt)
            response = self.get_normal_chatbot_response(prompt, chat_history)
            chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=response)
            ])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            self.chat_histories[session_id] = chat_history
            chat_response = {
                "response": response,
                "analysis": analysis,
                "is_serious": analysis.get("is_serious", False),
                "specialists": None
            }
            if analysis.get("is_serious", False):
                if self.vectorstore is None:
                    self.lazy_load_rag()
                if self.vectorstore is not None:
                    specialty = analysis.get("recommended_specialty", "general medicine")
                    specialists_data, _ = self.find_specialists_rag(specialty)
                    if specialists_data:
                        specialists = []
                        for doctor in specialists_data[:3]:
                            specialists.append({
                                "doctor_id": doctor.get('doctor_id', ''),
                                "name": doctor.get('name', ''),
                                "specialization": doctor.get('specialization', ''),
                                "category": doctor.get('category', ''),
                                "phone": doctor.get('phone', ''),
                                "experience": doctor.get('experience', 0)
                            })
                        chat_response["specialists"] = specialists
            return chat_response
        except Exception as e:
            return {"error": f"Error processing chat: {str(e)}"}
    
    def clear_chat(self, session_id: str = "default"):
        try:
            if session_id in self.chat_histories:
                self.chat_histories[session_id] = []
            return {"message": f"Chat history cleared for session: {session_id}"}
        except Exception as e:
            return {"error": f"Error clearing chat: {str(e)}"}
    
    def get_active_sessions(self):
        return {"active_sessions": list(self.chat_histories.keys())}
    
    def analyze_query_only(self, query: str):
        try:
            analysis = self.analyze_medical_query(query)
            return analysis
        except Exception as e:
            return {"error": f"Error analyzing query: {str(e)}"}
    
    def get_specialists_by_specialty(self, specialty: str, limit: int = 5):
        try:
            if self.vectorstore is None:
                self.lazy_load_rag()
            if self.vectorstore is None:
                return {"error": "Specialist database not available"}
            specialists_data, _ = self.find_specialists_rag(specialty, top_k=limit)
            specialists = []
            for doctor in specialists_data:
                specialists.append({
                    "doctor_id": doctor.get('doctor_id', ''),
                    "name": doctor.get('name', ''),
                    "specialization": doctor.get('specialization', ''),
                    "category": doctor.get('category', ''),
                    "phone": doctor.get('phone', ''),
                    "experience": doctor.get('experience', 0)
                })
            return {"specialists": specialists, "specialty": specialty}
        except Exception as e:
            return {"error": f"Error finding specialists: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Medical AI Chatbot", 
        page_icon="🩺", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">🩺 Medical AI Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Get medical assistance and specialist recommendations</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("📋 Chat Options")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_chat("streamlit_session")
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
        if 'chatbot' in st.session_state:
            sessions = st.session_state.chatbot.get_active_sessions()
            st.info(f"Active sessions: {len(sessions.get('active_sessions', []))}")
        st.markdown("---")
        st.markdown("### ℹ️ How to use:")
        st.markdown("""
        1. Ask about your health concerns
        2. Get medical advice and analysis
        3. Receive specialist recommendations for serious conditions
        4. Use clear, descriptive language about symptoms
        """)
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: This chatbot provides general medical information only. Always consult healthcare professionals for proper diagnosis and treatment.")
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing Medical AI Chatbot..."):
            st.session_state.chatbot = MedicalChatbot()
        st.success("✅ Medical AI Chatbot initialized!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("👋 Hello! I'm your Medical AI Assistant. How can I help you today?")
            st.markdown("You can ask me about:")
            st.markdown("- Symptoms and health concerns")
            st.markdown("- General medical advice")
            st.markdown("- Specialist recommendations")
            st.markdown("- Health and wellness questions")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "specialists" in message and message["specialists"]:
                st.markdown("### 👨‍⚕️ Recommended Specialists:")
                for i, doctor in enumerate(message["specialists"], 1):
                    st.info(f"""
**{i}. Dr. {doctor['name']}**
🏥 **Specialization:** {doctor['specialization']}
📞 **Phone:** {doctor['phone']}
📅 **Experience:** {doctor['experience']} years
🏷️ **Category:** {doctor.get('category', 'N/A')}
                    """)
            if "analysis" in message and message["analysis"]:
                analysis = message["analysis"]
                if analysis.get("is_serious"):
                    st.warning(f"""
**⚠️ Medical Analysis:**
- **Conditions detected:** {', '.join(analysis.get('detected_conditions', []))}
- **Recommended specialty:** {analysis.get('recommended_specialty', 'N/A')}
- **Explanation:** {analysis.get('explanation', 'N/A')}
                    """)
    if prompt := st.chat_input("Ask me about your health concerns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                response = st.session_state.chatbot.chat(prompt, "streamlit_session")
                if "error" not in response:
                    st.markdown(response["response"])
                    assistant_message = {
                        "role": "assistant", 
                        "content": response.get("response", "Error occurred")
                    }
                    if response.get("analysis") and response["analysis"].get("is_serious"):
                        analysis = response["analysis"]
                        st.warning(f"""
**⚠️ Medical Analysis:**
- **Conditions detected:** {', '.join(analysis.get('detected_conditions', []))}
- **Recommended specialty:** {analysis.get('recommended_specialty', 'N/A')}
- **Explanation:** {analysis.get('explanation', 'N/A')}
                        """)
                        assistant_message["analysis"] = analysis
                    if response.get("specialists"):
                        st.markdown("### 👨‍⚕️ Recommended Specialists:")
                        for i, doctor in enumerate(response["specialists"], 1):
                            st.info(f"""
**{i}. Dr. {doctor['name']}**
🏥 **Specialization:** {doctor['specialization']}
📞 **Phone:** {doctor['phone']}
📅 **Experience:** {doctor['experience']} years
🏷️ **Category:** {doctor.get('category', 'N/A')}
                            """)
                        assistant_message["specialists"] = response["specialists"]
                    st.session_state.messages.append(assistant_message)
                else:
                    error_msg = f"❌ Error: {response['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def command_line_interface():
    print("Medical AI Chatbot - Type 'quit' to exit")
    print("-" * 50)
    chatbot = MedicalChatbot()
    session_id = "cli_session"
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! Take care of your health!")
            break
        response = chatbot.chat(user_input, session_id)
        if "error" in response:
            print(f"Error: {response['error']}")
            continue
        print(f"\nBot: {response['response']}")
        if response.get("analysis"):
            analysis = response["analysis"]
            if analysis.get("is_serious"):
                print(f"\n⚠️  Medical Analysis:")
                print(f"Conditions detected: {analysis.get('detected_conditions', [])}")
                print(f"Recommended specialty: {analysis.get('recommended_specialty', 'N/A')}")
                print(f"Explanation: {analysis.get('explanation', 'N/A')}")
        if response.get("specialists"):
            print(f"\n👨‍⚕️ Recommended Specialists:")
            for i, doctor in enumerate(response["specialists"], 1):
                print(f"{i}. Dr. {doctor['name']} - {doctor['specialization']}")
                print(f"   Experience: {doctor['experience']} years | Phone: {doctor['phone']}")

if __name__ == "__main__":
    try:
        st.session_state
        main()
    except:
        command_line_interface()
