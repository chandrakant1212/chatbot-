import os
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize the LLM with Together AI API"""
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables")

    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=together_api_key,
        openai_api_base="https://api.together.xyz/v1",
        model="meta-llama/Llama-3-70b-chat-hf"
    )
    return llm

def create_vector_db():
    """Create vector database from PDF documents"""
    loader = DirectoryLoader(
        "data",
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vector_db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory='./chroma_db'
    )
    print("Chroma DB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    """Setup QA chain with custom prompt"""
    retriever = vector_db.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a compassionate mental health support assistant. Use the following context to provide helpful, empathetic responses.
        
        Guidelines:
        - Be supportive and understanding
        - Provide practical advice when appropriate
        - Always encourage professional help for serious concerns
        - Never provide medical diagnoses
        - Be empathetic and non-judgmental
        
        Context: {context}
        Question: {question}
        
        Answer:
        """
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Initialize components
print("Initializing Mental Health ChatBot...")
llm = initialize_llm()

db_path = "chroma_db"
if not os.path.exists(db_path):
    print("Creating vector database from documents...")
    if os.path.exists("data"):
        vector_db = create_vector_db()
    else:
        print("Warning: 'data' directory not found. Creating empty vector database.")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("Loading existing vector database...")
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
    )

qa_chain = setup_qa_chain(vector_db, llm)

# Custom CSS for modern mental health theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
    --healing-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --warm-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    --glass-bg: rgba(255, 255, 255, 0.25);
    --glass-border: rgba(255, 255, 255, 0.3);
    --text-primary: #2d3748;
    --text-light: #4a5568;
    --shadow-soft: 0 8px 32px rgba(31, 38, 135, 0.37);
    --shadow-glow: 0 0 30px rgba(102, 126, 234, 0.3);
}

* {
    font-family: 'Nunito', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    position: relative;
}

.chatbot {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: hidden !important;
    position: relative !important;
}

.chatbot::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--healing-gradient);
    z-index: 1;
}

.message.user {
    background: var(--primary-gradient) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    margin-left: auto !important;
    max-width: 75% !important;
    padding: 16px 20px !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    border: none !important;
}

.message.bot {
    background: rgba(255, 255, 255, 0.98) !important;
    color: #1a202c !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    max-width: 75% !important;
    padding: 16px 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    line-height: 1.6 !important;
    backdrop-filter: blur(10px) !important;
}

.message.bot * {
    color: #1a202c !important;
    font-weight: 500 !important;
}

textarea {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 20px !important;
    border: 1px solid var(--glass-border) !important;
    color: white !important;
    font-size: 16px !important;
}

textarea::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
}

button {
    background: var(--healing-gradient) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4) !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(79, 172, 254, 0.6) !important;
}

h1 {
    color: white !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
}

p {
    color: rgba(255, 255, 255, 0.9) !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    font-weight: 400 !important;
    font-size: 1.1rem !important;
}

footer {
    display: none !important;
}

@media (max-width: 768px) {
    .message.user, .message.bot {
        max-width: 90% !important;
        padding: 12px 16px !important;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    .chatbot {
        border-radius: 16px !important;
        margin: 10px !important;
    }
}
"""

def chatbot_response(message, history):
    """Generate chatbot response"""
    if not message.strip():
        return "Please enter a valid message"
    
    try:
        result = qa_chain({"query": message})
        response = result.get("result", "I couldn't generate a response.")
        
        # Add crisis resources for concerning keywords
        crisis_keywords = ["depressed", "sad", "suicide", "anxious", "hurt myself", "end it all", "can't go on"]
        if any(word in message.lower() for word in crisis_keywords):
            response += """

💜 **Important Resources:**
- **Crisis Text Line**: Text HOME to 741741 (US)
- **National Suicide Prevention Lifeline**: 988 (US)
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
- **Emergency Services**: 911 (US) or your local emergency number

You matter, and help is available. Please reach out to a mental health professional if you're struggling.
            """
            
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return "I'm sorry, I encountered an error. Please try again or consider speaking with a mental health professional if you need immediate support."

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="violet"), title="Mental Health Companion") as app:
    
    # Header
    with gr.Row():
        gr.Markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>🌸 Mental Health Companion 🌸</h1>
            <p>Your supportive AI wellness companion - Here to listen and support you</p>
        </div>
        """)
    
    # Disclaimer
    with gr.Row():
        gr.Markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin: 20px 0; border-left: 4px solid #4facfe;">
            <h3 style="color: white; margin-top: 0;">⚠️ Important Notice</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                This AI companion provides emotional support and general wellness information. 
                It is <strong>not a replacement</strong> for professional mental health care, therapy, or medical advice.
                If you're experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately.
            </p>
        </div>
        """)
    
    # Chat Interface
    chat_interface = gr.ChatInterface(
        fn=chatbot_response,
        examples=[
            "I'm feeling really anxious today 😰",
            "How can I practice mindfulness?",
            "What are some self-care tips?",
            "I'm having trouble sleeping",
            "How do I deal with stress?",
            "I feel overwhelmed with work",
            "Can you help me with breathing exercises?"
        ],
        title="",
        type="messages",
        description="",
    )
    
    # Footer
    with gr.Row():
        gr.Markdown("""
        <div style="text-align: center; margin-top: 30px; color: rgba(255,255,255,0.8);">
            <p>💚 Remember: You're not alone, and it's okay to ask for help. Taking care of your mental health is important.</p>
            <p><em>This chatbot uses AI to provide support but should not replace professional mental health services.</em></p>
        </div>
        """)

# Launch the application
if __name__ == "__main__":
      app.launch(
       share=True,
       server_name="0.0.0.0",
       server_port=7861,  # <- use a different unused port like 7861 or 8080
       show_error=True
   )
