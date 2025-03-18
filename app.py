import os
import io
import PyPDF2
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pyzbar.pyzbar import decode
from PIL import Image
import requests
from deep_translator import GoogleTranslator
import pandas as pd
import pyaudio
import wave
import speech_recognition as sr
# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

# Streamlit App
st.title("CartMapper")

# Initialize session state for the chain
if "chain" not in st.session_state:
    st.session_state.chain = None

# Language selection
language = st.radio("Select Language:", ("English", "Hindi", "Odia", "Bengali", "Tamil"))

# File uploader for PDF and CSV
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

# Option to scan QR code from camera or upload an image
qr_option = st.radio("QR Code Input Method:", ("Upload QR Image", "Scan QR from Camera"))


def download_pdf_from_url(url):
    """Download a PDF from a URL and return its content as bytes."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Check if the content is a valid PDF
            if response.content[:4] == b"%PDF":  # Check for PDF magic number
                return response.content
            else:
                st.error("The downloaded content is not a valid PDF.")
                return None
        else:
            st.error(f"Failed to download PDF. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while downloading the PDF: {e}")
        return None


def process_pdf(pdf_file):
    """Process a PDF file and return a list of documents."""
    try:
        data = []
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                data.append({
                    'page_number': page_num + 1,
                    'page_content': page_text
                })
        return [Document(page_content=page['page_content']) for page in data]
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Failed to process the PDF: {e}")
        return None


def process_csv(csv_file):
    """Process a CSV file and return a list of documents."""
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Convert each row into a Document object
        documents = []
        for index, row in df.iterrows():
            # Combine all columns into a single string
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(page_content=content))

        return documents
    except Exception as e:
        st.error(f"Failed to process the CSV file: {e}")
        return None


def setup_rag_chain(documents):
    """Set up the RAG (Retrieval-Augmented Generation) chain."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Set up API key
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create vector database with local persistence
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="huggingface-groq-rag",
        persist_directory="./chroma_db"  # Local directory to store the database
    )

    # Initialize LLM
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.environ['GROQ_API_KEY']
    )

    # Prompt for query reformulation
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant generating alternative query perspectives.
        Generate 5 different versions of the given question to improve document retrieval:
        Original question: {question}"""
    )

    # MultiQueryRetriever setup
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    # Define prompt template for answering questions
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


# Process QR code based on the selected option
if qr_option == "Upload QR Image":
    uploaded_qr = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"])
    if uploaded_qr:
        image = Image.open(uploaded_qr)
        decoded_objects = decode(image)
elif qr_option == "Scan QR from Camera":
    camera_image = st.camera_input("Scan QR Code")
    if camera_image:
        image = Image.open(camera_image)
        decoded_objects = decode(image)

# Decode QR code and process PDF
if 'decoded_objects' in locals() and decoded_objects:
    qr_data = decoded_objects[0].data.decode("utf-8")
    st.write(f"QR Code Data: {qr_data}")

    # Validate the QR data (ensure it's a URL)
    if qr_data.startswith("http://") or qr_data.startswith("https://"):
        # Download PDF from the URL in the QR code
        pdf_content = download_pdf_from_url(qr_data)

        if pdf_content:
            # Wrap the bytes in a BytesIO object
            pdf_file = io.BytesIO(pdf_content)

            # Process downloaded PDF
            documents = process_pdf(pdf_file)

            if documents:  # Ensure documents were processed successfully
                # Set up RAG chain
                st.session_state.chain = setup_rag_chain(documents)
    else:
        st.error("The QR code does not contain a valid URL.")
else:
    if qr_option == "Upload QR Image" and uploaded_qr:
        st.error("No QR code found in the uploaded image.")
    elif qr_option == "Scan QR from Camera" and camera_image:
        st.error("No QR code found in the camera image.")

# Process uploaded file (PDF or CSV)
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        # Process uploaded PDF
        documents = process_pdf(uploaded_file)
    elif uploaded_file.type == "text/csv":
        # Process uploaded CSV
        documents = process_csv(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or CSV file.")
        documents = None

    if documents:  # Ensure documents were processed successfully
        # Set up RAG chain
        st.session_state.chain = setup_rag_chain(documents)

# Function to record audio
def record_audio(filename="voice_input.wav", record_seconds=5, sample_rate=44100, chunk=1024):
    """Records audio from the microphone and saves it as a WAV file."""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    frames = []
    st.write("🎤 Recording... Please speak.")

    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    st.write("✅ Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    
    return filename

# Function to transcribe the recorded audio
def transcribe_audio(filename, lang="en"):
    """Converts speech from the audio file to text in the selected language."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language=lang)  # Transcribes in the selected language
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return None

# Detect input method
input_method = st.radio("Select Input Method:", ("Text Input", "Voice Input"))

query = ""

if input_method == "Voice Input":
    if st.button("🎤 Record Voice"):
        audio_filename = record_audio()
        lang_code = "en"  # Default English
        
        if language == "Hindi":
            lang_code = "hi-IN"
        elif language == "Odia":
            lang_code = "or-IN"
        elif language == "Bengali":
            lang_code = "bn-IN"
        elif language == "Tamil":
            lang_code = "ta-IN"  # Tamil transcription

        transcribed_text = transcribe_audio(audio_filename, lang=lang_code)
        
        if transcribed_text:
            st.session_state.transcribed_query = transcribed_text  # Store transcribed text
            st.write(f"**Transcribed Query:** {transcribed_text}")

# Use transcribed query if voice input was used
if input_method == "Voice Input" and "transcribed_query" in st.session_state:
    query = st.session_state.transcribed_query
else:
    # Text input
    if language == "Hindi":
        query = st.text_input("हिंदी में प्रश्न दर्ज करें:")
    elif language == "Odia":
        query = st.text_input("ଓଡ଼ିଆରେ ପ୍ରଶ୍ନ ପ୍ରବେଶ କରନ୍ତୁ:")
    elif language == "Bengali":
        query = st.text_input("বাংলায় প্রশ্ন লিখুন:")
    elif language == "Tamil":
        query = st.text_input("தமிழில் கேள்வியை உள்ளிடவும்:")
    else:
        query = st.text_input("Enter your question:")

# Process query when "Get Answer" is clicked
if st.button("Get Answer"):
    if query.strip() == "":
        st.error("Please enter or record a query before fetching an answer.")
    elif st.session_state.chain:
        # Translate input to English
        if language == "Hindi":
            translated_query = translator.translate(query, src='hi', dest='en')
        elif language == "Odia":
            translated_query = translator.translate(query, src='or', dest='en')
        elif language == "Bengali":
            translated_query = translator.translate(query, src='bn', dest='en')
        elif language == "Tamil":
            translated_query = translator.translate(query, src='ta', dest='en')
        else:
            translated_query = query

        # Invoke the RAG chain with the query
        result = st.session_state.chain.invoke(translated_query)

        # Translate output back to the selected language
        if language == "Hindi":
            result = translator.translate(result, src='en', dest='hi')
        elif language == "Odia":
            result = translator.translate(result, src='en', dest='or')
        elif language == "Bengali":
            result = translator.translate(result, src='en', dest='bn')
        elif language == "Tamil":
            result = translator.translate(result, src='en', dest='ta')

        st.write("### Answer:")
        st.write(result)
    else:
        st.error("No file has been processed yet. Please upload a PDF or CSV file or scan a QR code.")
