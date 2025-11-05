import os
import io
import pathlib
import math
import random
import time
import json
import tempfile
import base64
import numpy as np
import pandas as pd
import cv2
import pyaudio
import wave
import speech_recognition as sr
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from deep_translator import GoogleTranslator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import PyPDF2
import requests
import qrcode

# Google OAuth
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ==================== GOOGLE AUTH CONFIG ====================

CLIENT_SECRETS_FILE = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.profile",
          "https://www.googleapis.com/auth/userinfo.email"]
REDIRECT_URI = "http://localhost:8501/"  # ⚠️ Change this if deploying


def check_auth():
    """Show login if user not authenticated"""
    if "user" not in st.session_state:
        st.title("🔐 CartMapper Login")

        try:
            flow = Flow.from_client_secrets_file(
                client_secrets_file=CLIENT_SECRETS_FILE,
                scopes=SCOPES,
                redirect_uri=REDIRECT_URI
            )

            auth_url, state = flow.authorization_url(prompt="consent")
            st.session_state["oauth_state"] = state

            st.markdown(f"[👉 Continue with Google]({auth_url})")
        except FileNotFoundError:
            st.error(
                "❌ Google OAuth configuration file (client_secret.json) not found. Running without authentication.")
            st.session_state["user"] = {
                "name": "Guest User",
                "email": "guest@example.com",
                "picture": "https://via.placeholder.com/60"
            }
            st.info("Authentication bypassed - running as guest user")
            return True
        except Exception as e:
            st.error(f"❌ OAuth setup failed: {e}. Running without authentication.")
            st.session_state["user"] = {
                "name": "Guest User",
                "email": "guest@example.com",
                "picture": "https://via.placeholder.com/60"
            }
            return True

        st.stop()
    else:
        st.sidebar.success(f"✅ Logged in as {st.session_state['user']['email']}")
        return True


def handle_oauth_callback():
    """Handle redirect from Google after login"""
    query_params = st.query_params
    if "code" in query_params:
        try:
            flow = Flow.from_client_secrets_file(
                client_secrets_file=CLIENT_SECRETS_FILE,
                scopes=SCOPES,
                redirect_uri=REDIRECT_URI,
                state=st.session_state.get("oauth_state")
            )

            current_url = REDIRECT_URI + "?" + "&".join([f"{k}={v}" for k, v in query_params.items()])
            flow.fetch_token(authorization_response=current_url)

            creds = flow.credentials
            user_info = id_token.verify_oauth2_token(
                creds._id_token, google_requests.Request(), flow.client_config["client_id"]
            )

            st.session_state["user"] = {
                "name": user_info.get("name", "Google User"),
                "email": user_info.get("email", ""),
                "picture": user_info.get("picture", "")
            }

            st.query_params.clear()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")
            st.stop()


# ==================== STREAMLIT APP CONFIG ====================

st.set_page_config(page_title="CartMapper", layout="wide", page_icon="🛒")

# ==================== THEME & AESTHETIC STYLING ====================

st.markdown("""
<style>

:root {
  --primary-color: #ff6f61;  /* Coral pink */
  --accent-color: #ffd6a5;   /* Soft peach */
  --bg-gradient: linear-gradient(135deg, #f9f7f7 0%, #fff5f7 100%);
  --text-dark: #2f2f2f;
  --text-light: #6c757d;
  --card-bg: #ffffff;
  --border-radius: 18px;
  --shadow: 0 4px 14px rgba(0,0,0,0.08);
}

/* Background and typography */
html, body, [class*="css"]  {
  font-family: 'Poppins', sans-serif;
  background: var(--bg-gradient);
  color: var(--text-dark);
}

/* Title styling */
h1, h2, h3 {
  font-family: 'Playfair Display', serif;
  color: #222;
  text-shadow: 0 1px 0 rgba(0,0,0,0.05);
}

/* Streamlit sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fff5f7 0%, #fceef3 100%);
  color: #333;
  box-shadow: var(--shadow);
  border-radius: 0 20px 20px 0;
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h2 {
  color: #ff6f61;
}

/* Buttons */
div.stButton > button {
  background: var(--primary-color);
  color: white;
  font-weight: 600;
  border: none;
  border-radius: 12px;
  padding: 0.6em 1.2em;
  box-shadow: var(--shadow);
  transition: all 0.25s ease-in-out;
}
div.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(255,111,97,0.3);
  background: #ff826d;
}

/* Input fields and select boxes */
input, select, textarea, [data-baseweb="input"] {
  border-radius: 10px !important;
  border: 1px solid #ddd !important;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
  padding: 0.5em !important;
}
[data-testid="stTextInput"] label, [data-testid="stSelectbox"] label {
  font-weight: 600;
  color: var(--text-light);
}

/* Cards (like product, info, etc.) */
.stMarkdown, .stDataFrame, [data-testid="stVerticalBlock"] {
  border-radius: var(--border-radius);
  background: var(--card-bg);
  box-shadow: var(--shadow);
  padding: 1em;
  margin-top: 1em;
}

/* Expander styling */
details {
  background: #fffafc;
  border-radius: 15px;
  box-shadow: var(--shadow);
  border: none;
  padding: 0.5em 1em;
}
summary {
  font-weight: 600;
  color: var(--primary-color);
  font-size: 1.05rem;
}

/* Metric cards */
[data-testid="stMetricValue"] {
  color: var(--primary-color);
  font-weight: 700;
}
[data-testid="stMetricLabel"] {
  color: #555;
}

/* Product carousel container aesthetic */
.carousel-container {
  background: #fffafc;
  border-radius: 20px;
  padding: 20px;
  box-shadow: var(--shadow);
}
.product-card {
  background: white;
  border-radius: 15px;
  box-shadow: 0 4px 10px rgba(255,182,193,0.2);
  transition: all 0.3s ease;
}
.product-card:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 20px rgba(255,182,193,0.4);
}

/* Tabs and expanders */
[data-baseweb="tab"] {
  border-radius: 12px;
  background: #ffeef0;
  font-weight: 600;
  color: #333;
}

/* Footer / tips */
footer, .css-164nlkn {
  visibility: hidden;
}

</style>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# Handle OAuth first
handle_oauth_callback()
check_auth()

st.title("🛒 CartMapper - Document Analysis & Indoor Navigation")

st.markdown("""
<div style='
    background: linear-gradient(135deg, #ffd6a5, #ffcad4);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    color: #333;
    font-family: "Playfair Display", serif;
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
'>
✨ Welcome to <b>CartMapper</b> — Smart Shopping Meets AI.  
Explore, analyze, and navigate your world beautifully.
</div>
""", unsafe_allow_html=True)


# Sidebar user info
with st.sidebar:
    if st.session_state["user"]["picture"] != "https://via.placeholder.com/60":
        st.image(st.session_state["user"]["picture"], width=60)
    st.write(f"👋 Welcome, {st.session_state['user']['name']} ({st.session_state['user']['email']})")

    if st.button("🚪 Sign Out"):
        st.session_state.clear()
        st.rerun()

# ==================== SESSION STATE ====================

if "chain" not in st.session_state:
    st.session_state.chain = None
if "navigation_chain" not in st.session_state:
    st.session_state.navigation_chain = None
if "indoor_map" not in st.session_state:
    st.session_state.indoor_map = None
if "current_location" not in st.session_state:
    st.session_state.current_location = None
if "destinations" not in st.session_state:
    st.session_state.destinations = {}

# ==================== FEATURE SELECTION ====================

feature_mode = st.sidebar.selectbox(
    "Select Feature:",
    ["Document Analysis", "Indoor Navigation", "Both Features"]
)

language = st.radio("Select Language:", ("English", "Hindi", "Odia", "Bengali", "Tamil"))


# ==================== QR CODE FUNCTIONS WITH OPENCV ====================

def decode_qr_code(image):
    """Decode QR code using OpenCV"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            # Convert to RGB first if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

        # Ensure the image is in the correct format
        if not isinstance(image, np.ndarray):
            st.error("Invalid image format")
            return []

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.dtype == bool:
                image = image.astype(np.uint8) * 255
            else:
                image = image.astype(np.uint8)

        # Handle grayscale vs color images
        if len(image.shape) == 2:
            # Grayscale image - convert to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA image - convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                # RGB image - convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initialize QR code detector
        detector = cv2.QRCodeDetector()

        # Try to decode
        data, vertices, _ = detector.detectAndDecode(image)

        if data:
            # Return in pyzbar-compatible format
            return [type('obj', (object,), {'data': data.encode()})]
        else:
            # Try with grayscale conversion as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data, vertices, _ = detector.detectAndDecode(gray)

            if data:
                return [type('obj', (object,), {'data': data.encode()})]
            else:
                return []

    except Exception as e:
        st.error(f"QR code decoding error: {e}")
        return []


# ==================== UTILITY FUNCTIONS ====================

def get_translator(source='auto', target='en', max_retries=3):
    for attempt in range(max_retries):
        try:
            return GoogleTranslator(source=source, target=target)
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Translation service connection failed. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                st.error(
                    f"Could not connect to translation service after {max_retries} attempts. Proceeding without translation.")
                return None


translator = get_translator(source='auto', target='en')


def safe_translate(text, source_lang, target_lang, max_retries=2):
    if source_lang == target_lang:
        return text, True
    for attempt in range(max_retries):
        try:
            temp_translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = temp_translator.translate(text)
            if translated:
                return translated, True
            else:
                if attempt == max_retries - 1:
                    st.warning("Translation returned empty result. Using original text.")
                    return text, False
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Translation failed: {e}. Using original text.")
                return text, False
            else:
                st.info(f"Translation attempt {attempt + 1} failed. Retrying...")
                time.sleep(1)
    return text, False


# ==================== DOCUMENT PROCESSING ====================

def download_pdf_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if response.content[:4] == b"%PDF":
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
    try:
        data = []
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                data.append({
                    'page_number': page_num + 1,
                    'page_content': page_text
                })
        return [Document(page_content=page['page_content']) for page in data]
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Failed to process the PDF: {e}")
        return None


def process_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        documents = []
        for index, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(page_content=content))
        return documents
    except Exception as e:
        st.error(f"Failed to process the CSV file: {e}")
        return None


def setup_rag_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except KeyError:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in secrets.toml or environment variables.")
            return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="huggingface-groq-rag",
            persist_directory="./chroma_db"
        )

        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY
        )

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant generating alternative query perspectives.
            Generate 5 different versions of the given question to improve document retrieval:
            Original question: {question}"""
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        return chain

    except Exception as e:
        st.warning(f"Using simple text-based retrieval due to embedding issues: {e}")

        st.session_state.documents = chunks

        def simple_retriever(query):
            query_words = query.lower().split()
            scored_docs = []

            for doc in chunks:
                content = doc.page_content.lower()
                score = sum(1 for word in query_words if word in content)
                if score > 0:
                    scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:3]]

        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY
        )

        template = """Answer the question based on the following context:
        {context}

        Question: {question}

        If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                {"context": lambda x: "\n\n".join([doc.page_content for doc in simple_retriever(x["question"])]),
                 "question": lambda x: x["question"]}
                | prompt
                | llm
                | StrOutputParser()
        )
        return chain


# ==================== AUDIO FUNCTIONS ====================

def record_audio(filename="voice_input.wav", record_seconds=5, sample_rate=44100, chunk=1024):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk)
        frames = []
        st.write("Recording... Please speak.")
        for _ in range(0, int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        st.write("Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return filename
    except Exception as e:
        st.error(f"Audio recording failed: {e}")
        return None


def transcribe_audio(filename, lang="en"):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio, language=lang)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"Audio transcription failed: {e}")
        return None


def text_to_speech(text, lang_code="en", max_retries=2):
    for attempt in range(max_retries):
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_filename = fp.name
                tts.save(temp_filename)
            with open(temp_filename, "rb") as audio_file:
                audio_bytes = audio_file.read()
            os.remove(temp_filename)
            return audio_bytes, True
        except Exception as e:
            if lang_code != "en" and attempt == 0:
                st.warning(f"Failed to generate speech in {lang_code}. Trying English as fallback...")
                lang_code = "en"
            elif attempt == max_retries - 1:
                st.error(f"Failed to generate speech: {e}")
                return None, False
    return None, False


def get_tts_language_code(language):
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Odia": "or",
        "Bengali": "bn",
        "Tamil": "ta"
    }
    return language_codes.get(language, "en")


# ==================== INDOOR NAVIGATION CORE ====================

class IndoorMap:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.locations = {}
        self.obstacles = []
        self.paths = []
        self.products = {}

    def add_location(self, name, x, y, description=""):
        self.locations[name] = {'x': x, 'y': y, 'description': description}

    def add_obstacle(self, x1, y1, x2, y2):
        self.obstacles.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    def calculate_distance(self, loc1, loc2):
        if loc1 in self.locations and loc2 in self.locations:
            x1, y1 = self.locations[loc1]['x'], self.locations[loc1]['y']
            x2, y2 = self.locations[loc2]['x'], self.locations[loc2]['y']
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return float('inf')

    def find_path(self, start, end):
        if start not in self.locations or end not in self.locations:
            return None
        start_pos = self.locations[start]
        end_pos = self.locations[end]
        path = [
            {'x': start_pos['x'], 'y': start_pos['y'], 'location': start},
            {'x': end_pos['x'], 'y': end_pos['y'], 'location': end}
        ]
        return path

    def find_product_location(self, product_name):
        product_name_lower = product_name.lower()
        for location, products in self.products.items():
            for product in products:
                if product_name_lower in product['name'].lower():
                    return location
        return None

    def get_location_products(self, location_name):
        return self.products.get(location_name, [])

    def get_nearest_locations(self, current_location, max_distance=100):
        if current_location not in self.locations:
            return []
        current_pos = self.locations[current_location]
        nearby = []
        for name, pos in self.locations.items():
            if name != current_location:
                distance = math.sqrt((pos['x'] - current_pos['x']) ** 2 + (pos['y'] - current_pos['y']) ** 2)
                if distance <= max_distance:
                    nearby.append({'name': name, 'distance': distance, 'description': pos['description']})
        return sorted(nearby, key=lambda x: x['distance'])

    def get_products_by_type(self, product_type):
        filtered_products = []
        for location, products in self.products.items():
            for product in products:
                if product.get('type', '').lower() == product_type.lower():
                    filtered_products.append({'product': product, 'location': location})
        return filtered_products

    def get_products_by_price_range(self, min_price, max_price):
        filtered_products = []
        for location, products in self.products.items():
            for product in products:
                price = product.get('price', 0)
                try:
                    price = float(price)
                    if min_price <= price <= max_price:
                        filtered_products.append({'product': product, 'location': location})
                except (ValueError, TypeError):
                    continue
        return sorted(filtered_products, key=lambda x: x['product'].get('price', 0))

    def search_products(self, search_term):
        search_term = search_term.lower()
        results = []
        for location, products in self.products.items():
            for product in products:
                product_name = product.get('name', '').lower()
                product_category = product.get('category', '').lower()
                if (search_term in product_name or search_term in product_category):
                    results.append({
                        'product': product,
                        'location': location,
                        'match_score': (2 if search_term in product_name else 1)
                    })
        return sorted(results, key=lambda x: (-x['match_score'], x['product']['name']))

    def generate_visual_map(self, current_location=None, destination=None, path=None):
        img = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(img)

        for obstacle in self.obstacles:
            draw.rectangle([obstacle['x1'], obstacle['y1'], obstacle['x2'], obstacle['y2']],
                           fill='gray', outline='black')

        for name, loc in self.locations.items():
            color = 'blue'
            if name == current_location:
                color = 'green'
            elif name == destination:
                color = 'red'
            radius = 15
            draw.ellipse([loc['x'] - radius, loc['y'] - radius,
                          loc['x'] + radius, loc['y'] + radius],
                         fill=color, outline='black')
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            draw.text((loc['x'] + 20, loc['y']), name, fill='black', font=font)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                draw.line([path[i]['x'], path[i]['y'], path[i + 1]['x'], path[i + 1]['y']],
                          fill='red', width=3)
        return img


# ==================== LOCATION DETECTION ====================

def get_user_location():
    """Simulate GPS-like location detection for indoor navigation"""
    time.sleep(1)
    inside_probability = 0.85
    is_inside = random.random() < inside_probability
    accuracy = random.randint(3, 15)

    location_data = {
        'inside_store': is_inside,
        'accuracy': accuracy,
        'timestamp': time.time(),
        'confidence': random.uniform(0.7, 0.95),
        'detected_floor': 'Ground Floor' if is_inside else 'Unknown',
        'signal_strength': random.uniform(0.6, 1.0) if is_inside else random.uniform(0.1, 0.4)
    }

    return location_data


# ==================== SHOP PROFILE & QR NAV ====================

SHOP_PROFILE = {
    "shop_id": "shop_demo_001",
    "store_width_cm": 3000,
    "store_height_cm": 2000,
    "anchors": [
        {"anchor_id": "A1", "name": "A1 Entrance", "x": 100, "y": 1850},
        {"anchor_id": "A2", "name": "A2 Aisle Left", "x": 500, "y": 1200},
        {"anchor_id": "A3", "name": "A3 Aisle Right", "x": 2500, "y": 1200},
        {"anchor_id": "A4", "name": "A4 Electronics", "x": 2200, "y": 500},
        {"anchor_id": "A5", "name": "A5 Food Court", "x": 600, "y": 500},
        {"anchor_id": "A6", "name": "A6 Checkout", "x": 1500, "y": 150},
    ]
}


def scale_to_canvas(x_cm, y_cm, map_width_px=800, map_height_px=600,
                    store_width_cm=3000, store_height_cm=2000):
    x_px = int((x_cm / store_width_cm) * map_width_px)
    y_px = int((y_cm / store_height_cm) * map_height_px)
    return x_px, y_px


def add_shop_anchors_to_map(indoor_map, shop_profile=SHOP_PROFILE):
    for a in shop_profile["anchors"]:
        ax, ay = scale_to_canvas(
            a["x"], a["y"],
            map_width_px=indoor_map.width,
            map_height_px=indoor_map.height,
            store_width_cm=shop_profile["store_width_cm"],
            store_height_cm=shop_profile["store_height_cm"]
        )
        indoor_map.add_location(a["name"], ax, ay, f"Anchor {a['anchor_id']} ({a['name']})")


def parse_qr_anchor_payload(qr_text):
    try:
        payload = json.loads(qr_text)
        if payload.get("type") != "anchor":
            return None, "This QR is not an anchor marker."
        if payload.get("shop_id") != SHOP_PROFILE["shop_id"]:
            return None, f"This marker belongs to shop {payload.get('shop_id')}. Switch shop?"
        for k in ["anchor_id", "name", "x", "y"]:
            if k not in payload:
                return None, f"Missing field: {k}"
        return payload, None
    except Exception as e:
        return None, f"Bad QR data: {e}"


def make_qr_png_bytes(text: str) -> bytes:
    img = qrcode.make(text)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ==================== MAP CREATION ====================

def create_dynamic_store_map(store_df):
    mall_map = IndoorMap(800, 600)
    mall_map.add_location("Entrance", 400, 550, "Main entrance to the store")

    # 🔍 Automatically detect column names (works for "Price (INR)")
    name_col = next((c for c in store_df.columns if "name" in c.lower() or "product" in c.lower()), None)
    category_col = next((c for c in store_df.columns if "category" in c.lower()), None)
    price_col = next((c for c in store_df.columns if "price" in c.lower()), None)
    type_col = next((c for c in store_df.columns if "veg" in c.lower() or "type" in c.lower()), None)
    stock_col = next((c for c in store_df.columns if "stock" in c.lower() or "qty" in c.lower()), None)

    # 🗺️ Create store layout dynamically by categories
    if category_col:
        categories = store_df[category_col].fillna("General").unique()
        positions = [
            (150, 150), (650, 150), (150, 300), (650, 300),
            (150, 450), (650, 450), (400, 200), (400, 350)
        ]
        for i, category in enumerate(categories):
            if i < len(positions):
                x, y = positions[i]
                mall_map.add_location(f"{category} Section", x, y, f"Section for {category} products")
    else:
        mall_map.add_location("General Section", 400, 300, "General products area")

    mall_map.add_location("Checkout", 400, 500, "Billing counter")
    mall_map.add_location("Exit", 400, 50, "Store exit")

    mall_map.products = {}

    # 🧮 Helper to clean and parse price
    def parse_price(value):
        if pd.isna(value):
            return "N/A"
        try:
            s = str(value).replace(",", "").replace("₹", "").strip()
            return float(s)
        except Exception:
            return "N/A"

    # 🏷️ Create product entries
    for i, row in store_df.iterrows():
        product = {
            "name": str(row.get(name_col, f"Product {i + 1}")),
            "category": str(row.get(category_col, "General")) if category_col else "General",
            "price": parse_price(row.get(price_col)) if price_col else "N/A",
            "type": str(row.get(type_col, "Unknown")) if type_col else "Unknown"
        }
        if stock_col:
            product["stock"] = str(row.get(stock_col))

        section_name = f"{product['category']} Section" if category_col else "General Section"
        mall_map.products.setdefault(section_name, []).append(product)

    return mall_map


# ==================== DEFAULT SAMPLE MALL MAP (used if no CSV uploaded) ====================

def create_sample_mall_map():
    mall_map = IndoorMap(800, 600)
    mall_map.add_location("Entrance", 400, 550, "Main entrance to the mall")
    mall_map.add_location("Food Court", 200, 200, "Dining area with multiple restaurants")
    mall_map.add_location("Electronics Store", 600, 200, "Latest gadgets and electronics")
    mall_map.add_location("Clothing Store", 200, 400, "Fashion and apparel")
    mall_map.add_location("Grocery Store", 600, 400, "Supermarket for daily needs")
    mall_map.add_location("Pharmacy", 400, 300, "Medicine and healthcare products")
    mall_map.add_location("ATM", 100, 350, "Cash withdrawal point")
    mall_map.add_location("Restroom", 700, 350, "Public facilities")
    mall_map.add_location("Information Desk", 400, 450, "Help and information center")
    mall_map.add_location("Parking", 50, 550, "Vehicle parking area")

    mall_map.add_obstacle(350, 250, 450, 350)
    mall_map.add_obstacle(0, 0, 800, 50)
    mall_map.add_obstacle(0, 0, 50, 600)
    mall_map.add_obstacle(750, 0, 800, 600)
    mall_map.add_obstacle(0, 550, 800, 600)

    # 🛒 Sample products for default map
    mall_map.products = {
        "Grocery Store": [
            {"name": "Basmati Rice", "category": "Grains", "price": 120, "type": "Veg"},
            {"name": "Milk", "category": "Dairy", "price": 55, "type": "Veg"},
            {"name": "Bread", "category": "Bakery", "price": 25, "type": "Veg"},
            {"name": "Tomatoes", "category": "Vegetables", "price": 40, "type": "Veg"},
        ],
        "Food Court": [
            {"name": "Pizza", "category": "Fast Food", "price": 250, "type": "Veg"},
            {"name": "Burger", "category": "Fast Food", "price": 180, "type": "Non-Veg"},
            {"name": "Coffee", "category": "Beverages", "price": 80, "type": "Veg"},
        ],
        "Electronics Store": [
            {"name": "Smartphone", "category": "Electronics", "price": 15000, "type": "NA"},
            {"name": "Headphones", "category": "Electronics", "price": 2000, "type": "NA"},
        ]
    }

    return mall_map


# ==================== SETUP NAVIGATION CHAIN ====================

def setup_navigation_chain():
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except KeyError:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in secrets.toml or environment variables.")
            return None

    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

    navigation_template = """You are an intelligent indoor navigation assistant for a modern store/supermarket. 
Available locations: {locations}
Current location: {current_location}
Products available: {products_info}
User query: {question}

Instructions:
- Provide clear, helpful navigation assistance
- If asked for directions, give step-by-step walking instructions
- If asked about products, mention location, price, and type when available
- For shopping lists, suggest the most efficient route
- Be conversational and friendly
- Include relevant product recommendations when appropriate
- Mention distances when helpful
- If searching for products, list the best matches with locations

Response format:
- Use bullet points for directions
- Include product prices when mentioned
- Suggest alternatives if exact product not found
- Be concise but informative
Response:"""

    prompt = ChatPromptTemplate.from_template(navigation_template)
    chain = (prompt | llm | StrOutputParser())
    return chain


# ==================== FEATURE: DOCUMENTS ====================

if feature_mode in ["Document Analysis", "Both Features"]:
    st.header("Document Analysis")

    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

    qr_option = st.radio("QR Code Input Method:", ("Upload QR Image", "Scan QR from Camera"))
    decoded_objects = []

    if qr_option == "Upload QR Image":
        uploaded_qr = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"], key="qr_doc")
        if uploaded_qr:
            image = Image.open(uploaded_qr)
            decoded_objects = decode_qr_code(image)
    elif qr_option == "Scan QR from Camera":
        camera_image = st.camera_input("Scan QR Code")
        if camera_image:
            image = Image.open(camera_image)
            decoded_objects = decode_qr_code(image)

    if decoded_objects:
        qr_data = decoded_objects[0].data.decode("utf-8")
        st.write(f"QR Code Data: {qr_data}")
        if qr_data.startswith("http://") or qr_data.startswith("https://"):
            pdf_content = download_pdf_from_url(qr_data)
            if pdf_content:
                pdf_file = io.BytesIO(pdf_content)
                documents = process_pdf(pdf_file)
                if documents:
                    st.session_state.chain = setup_rag_chain(documents)
        else:
            st.error("The QR code does not contain a valid URL.")
    else:
        if qr_option == "Upload QR Image" and 'uploaded_qr' in locals() and uploaded_qr:
            st.error("No QR code found in the uploaded image.")
        elif qr_option == "Scan QR from Camera" and 'camera_image' in locals() and camera_image:
            st.error("No QR code found in the camera image.")

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            documents = process_pdf(uploaded_file)
        elif uploaded_file.type == "text/csv":
            documents = process_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF or CSV file.")
            documents = None
        if documents:
            st.session_state.chain = setup_rag_chain(documents)

# ==================== FEATURE: INDOOR NAV & PRODUCTS ====================

if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.header("Indoor Navigation & Product Finder")

    store_csv = st.file_uploader("Upload Store Inventory (CSV)", type=["csv"], key="store_csv")

    if store_csv is not None:
        store_df = pd.read_csv(store_csv)
        st.session_state.indoor_map = create_dynamic_store_map(store_df)
        add_shop_anchors_to_map(st.session_state.indoor_map, SHOP_PROFILE)
        st.session_state.navigation_chain = setup_navigation_chain()

        # ==================== SAFE CSV UPLOAD HANDLING ====================

        if store_csv is not None:
            # Validate the uploaded file type
            if not store_csv.name.lower().endswith(".csv"):
                st.error("Please upload a valid CSV file (not PDF, TXT, or other formats).")
                st.stop()

            try:
                # Reset the file pointer to start (important!)
                store_csv.seek(0)

                # Try to read the CSV file safely
                store_df = pd.read_csv(store_csv)

                # Validate the CSV structure
                if store_df.empty or len(store_df.columns) == 0:
                    st.error("⚠️ The uploaded CSV file appears empty or has no columns.")
                    st.stop()

            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded CSV file is empty or corrupted.")
                st.stop()

            except Exception as e:
                st.error(f"❌ Could not read the uploaded CSV file: {e}")
                st.stop()

            # Continue only if everything above succeeded
            st.session_state.indoor_map = create_dynamic_store_map(store_df)
            add_shop_anchors_to_map(st.session_state.indoor_map, SHOP_PROFILE)
            st.session_state.navigation_chain = setup_navigation_chain()

        # ==================== PRODUCT CAROUSEL WITH FILTER ====================

        # ==================== PRODUCT CAROUSEL WITH FILTER ====================

        # ==================== PRODUCT CAROUSEL WITH FILTER ====================

        if hasattr(st.session_state.indoor_map, "products") and st.session_state.indoor_map.products:
            st.markdown("## 🛍️ Product Showcase")

            # Combine all products across sections
            all_products = []
            for section, products in st.session_state.indoor_map.products.items():
                for p in products:
                    all_products.append({
                        "name": p.get("name", "Unnamed"),
                        "category": p.get("category", "General"),
                        "price": p.get("price", "N/A"),
                        "type": p.get("type", "Unknown"),
                        "section": section
                    })

            # Filters
            categories = sorted(set([p["category"] for p in all_products if p.get("category")]))
            types = sorted(set([p["type"] for p in all_products if p.get("type")]))
            col1, col2 = st.columns(2)
            with col1:
                selected_category = st.selectbox("🔎 Filter by Category", ["All"] + categories, index=0)
            with col2:
                selected_type = st.selectbox("🥗 Filter by Type", ["All"] + types, index=0)

            # Filter logic
            filtered_products = [
                p for p in all_products
                if (selected_category == "All" or p["category"] == selected_category)
                   and (selected_type == "All" or p["type"] == selected_type)
            ]

            # Build HTML carousel
            html_parts = []
            html_parts.append("""
            <style>
            /* ==================== PASTEL PRODUCT SHOWCASE ==================== */

            .showcase-wrapper {
                overflow: hidden;
                width: 100%;
                position: relative;
                background: linear-gradient(135deg, #fff9fb 0%, #fffefc 100%);
                border-radius: 22px;
                box-shadow: 0 6px 16px rgba(0,0,0,0.05);
                padding: 1.5rem 0;
                border: none;
            }

            /* 🌸 Smooth, slow scroll */
            .showcase-carousel-container {
                display: flex;
                gap: 1.3rem;
                animation: showcaseScroll 150s linear infinite;
                width: max-content;
            }

            @keyframes showcaseScroll {
                0% { transform: translateX(0); }
                100% { transform: translateX(-50%); }
            }

            /* 🩰 Soft pastel product cards */
            .showcase-card {
                flex: 0 0 260px;
                background: linear-gradient(180deg, #ffffff, #fff8fb);
                border-radius: 18px;
                border: 1px solid rgba(255,182,193,0.25);
                box-shadow: 0 6px 14px rgba(255,150,170,0.12);
                padding: 1rem;
                text-align: center;
                transition: transform 0.45s ease, box-shadow 0.45s ease;

            }

            .showcase-card:hover {
                transform: translateY(-6px) scale(1.02);
                box-shadow: 0 10px 24px rgba(255,160,180,0.25);
                border-color: rgba(255,150,170,0.4);
                background: linear-gradient(180deg, #fff, #fff3f6);
            }

            .showcase-title { font-weight: 600; font-size: 1.05rem; color: #222; margin-bottom: 0.25rem; }
            .showcase-category { font-size: 0.9rem; color: #777; margin-bottom: 0.4rem; }
            .showcase-price { font-weight: 700; color: #2e7d32; margin-bottom: 0.3rem; }
            .showcase-type {
                font-size: 0.8rem; color: #333; background: #f7f7f7;
                padding: 3px 8px; border-radius: 6px; display: inline-block; margin-bottom: 0.4rem;
            }
            .showcase-section {
                font-size: 0.75rem; color: #444; background: #eef6ff;
                border-radius: 6px; padding: 3px 6px; display: inline-block;
            }

            /* Pause animation on hover */
            .showcase-wrapper:hover .showcase-carousel-container { animation-play-state: paused; }

            /* Hide scrollbar */
            .showcase-carousel-container::-webkit-scrollbar { display: none; }

            </style>

            <div class="showcase-wrapper">
                <div class="showcase-carousel-container">
            """)

            for p in filtered_products:
                # 🔁 Duplicate for seamless scrolling
                for p in filtered_products:
                    price_display = p["price"]
                    if isinstance(price_display, (int, float)):
                        price_display = f"₹{price_display:.2f}"
                    html_parts.append(f"""
                    <<div class="showcase-card">
    <div class="showcase-title">{p['name']}</div>
    <div class="showcase-category">{p['category']}</div>
    <div class="showcase-price">{p['price'] if isinstance(p['price'], str) else f"₹{p['price']:.2f}"}</div>
    <div class="showcase-type">{p['type']}</div>
    <div class="showcase-section">📍 {p['section']}</div>
</div>

                    """)

            html_parts.append("</div></div>")
            carousel_html = "\n".join(html_parts)

            if not filtered_products:
                st.warning("No products match the selected filters.")
            else:
                import streamlit.components.v1 as components

                components.html(
                    f"<div style='padding:10px;background:linear-gradient(135deg, #fff9fb 0%, #fffefc 100%); border-radius:20px; box-shadow:0 4px 12px rgba(255,182,193,0.25);'>{carousel_html}</div>",
                    height=480, scrolling=False
                )

        # ==================== PRODUCT CAROUSEL (Offline Mode) ====================
        st.markdown("### 🛍️ Product Carousel Preview")

        if hasattr(st.session_state.indoor_map, "products") and st.session_state.indoor_map.products:
            all_products = []
            for location, products in st.session_state.indoor_map.products.items():
                for p in products:
                    item = {
                        "name": p.get("name", "Unnamed Product"),
                        "category": p.get("category", "General"),
                        "price": p.get("price", "N/A"),
                        "type": p.get("type", "Unknown"),
                        "location": location
                    }
                    all_products.append(item)

            if all_products:
                carousel_items = ""
                for prod in all_products[:10]:  # limit to 10 for speed
                    # Placeholder product image (no internet required)
                    img_text = prod['name'].replace(' ', '+')
                    img_url = f"https://via.placeholder.com/200x150.png?text={img_text}"

                    carousel_items += f"""
                    <div class="carousel-item">
                        <img src="{img_url}" alt="{prod['name']}" />
                        <div class="product-info">
                            <h4>{prod['name']}</h4>
                            <p><b>Category:</b> {prod['category']}</p>
                            <p><b>Price:</b> ₹{prod['price']}</p>
                            <p><b>Section:</b> {prod['location']}</p>
                        </div>
                    </div>
                    """

                carousel_html = f"""
                <div class="carousel-container">
                    <div class="carousel-track">
                        {carousel_items}
                    </div>
                </div>

                <style>
                .carousel-container {{
                    width: 100%;
                    overflow: hidden;
                    position: relative;
                    background: #f9f9f9;
                    border-radius: 15px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    padding: 10px;
                    margin-bottom: 20px;
                }}
                .carousel-track {{
                    display: flex;
                    width: max-content;
                    animation: scroll 30s linear infinite;
                }}
                .carousel-item {{
                    flex: 0 0 auto;
                    width: 200px;
                    margin: 0 15px;
                    background: white;
                    border-radius: 12px;
                    padding: 10px;
                    text-align: center;
                    transition: transform 0.3s;
                    border: 1px solid #eee;
                }}
                .carousel-item:hover {{
                    transform: scale(1.05);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .carousel-item img {{
                    width: 100%;
                    height: 150px;
                    object-fit: cover;
                    border-radius: 8px;
                }}
                .product-info {{
                    font-size: 13px;
                    margin-top: 8px;
                    color: #333;
                }}
                .product-info h4 {{
                    font-size: 14px;
                    margin-bottom: 5px;
                    color: #111;
                }}
                .product-card:hover {{
                    transform: translateY(-6px);
                    box-shadow: 0 10px 25px rgba(50, 205, 50, 0.25);
                    background: linear-gradient(180deg, #ffffff, #f7fdf8);
                    }}
                @keyframes scroll {{
                    0% {{ transform: translateX(0); }}
                    100% {{ transform: translateX(-50%); }}
                }}


                </style>
                """
                st.components.v1.html(carousel_html, height=280, scrolling=False)
            else:
                st.info("No products found to display in the carousel yet.")
        else:
            st.info("Upload a store inventory CSV or scan a QR that links to a CSV to show products here.")

        with st.expander("Store Inventory Overview"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Products", len(store_df))
            with col2:
                st.metric("Locations", len(st.session_state.indoor_map.locations))
            with col3:
                veg_count = len(store_df[store_df.get('Veg/Non-Veg', 'Veg') == 'Veg'])
                st.metric("Veg Products", veg_count)
            with col4:
                categories = store_df['Category'].nunique() if 'Category' in store_df else 0
                st.metric("Categories", categories)
            st.dataframe(store_df.head())

    elif st.session_state.indoor_map is None:
        st.info("Upload a store inventory CSV to generate the store layout, or use the default supermarket layout.")
        if st.button("Use Sample Supermarket Layout"):
            st.session_state.indoor_map = create_sample_mall_map()
            add_shop_anchors_to_map(st.session_state.indoor_map, SHOP_PROFILE)
            st.session_state.navigation_chain = setup_navigation_chain()

    if st.session_state.indoor_map is not None:
        location_col1, location_col2 = st.columns([2, 1])

        with location_col2:
            st.subheader("Location Services (QR Preferred)")

            qr_tab1, qr_tab2, qr_tab3 = st.tabs(["Upload Anchor QR", "Scan via Camera", "Auto Detect"])
            decoded_anchor = []

            with qr_tab1:
                uploaded_qr = st.file_uploader("Upload an anchor QR (A1–A6)", type=["png", "jpg", "jpeg"],
                                               key="qr_nav_upload")
                if uploaded_qr:
                    try:
                        image = Image.open(uploaded_qr)
                        decoded_anchor = decode_qr_code(image)
                    except Exception as e:
                        st.error(f"Could not read image: {e}")

            with qr_tab2:
                camera_image = st.camera_input("Point at A1–A6 QR")
                if camera_image:
                    try:
                        image = Image.open(camera_image)
                        decoded_anchor = decode_qr_code(image)
                    except Exception as e:
                        st.error(f"Camera image error: {e}")

            with qr_tab3:
                if st.button("Detect My Location"):
                    user_location = get_user_location()
                    if user_location['inside_store']:
                        st.success("Location detected: Inside store")
                        st.info(f"Accuracy: ±{user_location['accuracy']} meters")
                        st.info(f"Confidence: {user_location['confidence']:.1%}")
                        st.info(f"Signal Strength: {user_location['signal_strength']:.1%}")
                        if not st.session_state.current_location:
                            st.session_state.current_location = "Entrance"
                            st.success("Set current location to: Entrance")
                    else:
                        st.warning("You appear to be outside the store")

            if decoded_anchor:
                qr_text = decoded_anchor[0].data.decode("utf-8")
                payload, err = parse_qr_anchor_payload(qr_text)
                if err:
                    st.warning(err)
                else:
                    x_px, y_px = scale_to_canvas(
                        payload["x"], payload["y"],
                        map_width_px=st.session_state.indoor_map.width,
                        map_height_px=st.session_state.indoor_map.height,
                        store_width_cm=SHOP_PROFILE["store_width_cm"],
                        store_height_cm=SHOP_PROFILE["store_height_cm"]
                    )
                    st.session_state.indoor_map.add_location(payload["name"], x_px, y_px,
                                                             f"Anchor {payload['anchor_id']} ({payload['name']})")
                    st.session_state.current_location = payload["name"]
                    st.success(f"Position fixed via QR: **{payload['name']}**")

            st.caption("Manual fallback (use only if QR unavailable):")
            location_names = list(st.session_state.indoor_map.locations.keys())
            current_location = st.selectbox(
                "Manually select location:",
                ["(Prefer QR/Auto)"] + location_names,
                index=0 if st.session_state.current_location is None else
                location_names.index(st.session_state.current_location) + 1
                if st.session_state.current_location in location_names else 0
            )
            if current_location != "(Prefer QR/Auto)":
                st.session_state.current_location = current_location

            if st.session_state.current_location:
                st.success(f"Current: **{st.session_state.current_location}**")
            else:
                st.info("No location fixed yet. Scan any A1–A6 anchor QR or use auto-detect.")

            with st.expander("Download: Anchor QR Codes (A1–A6)"):
                st.write("These are ready-to-print PNGs. Each encodes a JSON payload for this shop.")
                for a in SHOP_PROFILE["anchors"]:
                    payload = {
                        "type": "anchor",
                        "shop_id": SHOP_PROFILE["shop_id"],
                        "anchor_id": a["anchor_id"],
                        "name": a["name"],
                        "x": a["x"],
                        "y": a["y"],
                        "v": 1
                    }
                    payload_text = json.dumps(payload, separators=(',', ':'))
                    png_bytes = make_qr_png_bytes(payload_text)
                    st.image(png_bytes, caption=f"{a['anchor_id']} – {a['name']}", use_column_width=True)
                    st.download_button(
                        label=f"Download {a['anchor_id']} QR (PNG)",
                        data=png_bytes,
                        file_name=f"{a['anchor_id']}_{a['name'].replace(' ', '_')}.png",
                        mime="image/png"
                    )
                    st.code(payload_text, language="json")

        with location_col1:
            st.subheader("Product Search")
            search_query = st.text_input("Search for products:", placeholder="e.g., Basmati Rice, Tomato, Chicken")

            if search_query:
                found_products = []
                locations_with_products = []

                if hasattr(st.session_state.indoor_map, 'products'):
                    for location, products in st.session_state.indoor_map.products.items():
                        for product in products:
                            if search_query.lower() in product['name'].lower():
                                found_products.append({'product': product, 'location': location})
                                if location not in locations_with_products:
                                    locations_with_products.append(location)

                if found_products:
                    st.success(f"Found {len(found_products)} matching product(s)")
                    for item in found_products[:5]:
                        product = item['product']
                        location = item['location']

                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{product['name']}** - {location}")
                            st.caption(f"{product['category']} | {product['type']}")
                        with col2:
                            st.write(f"₹{product['price']}")
                        with col3:
                            if st.button(f"Go", key=f"goto_{product['name']}"):
                                if st.session_state.current_location and location in st.session_state.indoor_map.locations:
                                    path = st.session_state.indoor_map.find_path(st.session_state.current_location,
                                                                                 location)
                                    distance = st.session_state.indoor_map.calculate_distance(
                                        st.session_state.current_location, location)
                                    st.info(f"Route to {location}: ~{distance:.0f} units")
                                    nav_query = f"Give me directions from {st.session_state.current_location} to {location} to find {product['name']}"
                                    locations_info = {name: info['description'] for name, info in
                                                      st.session_state.indoor_map.locations.items()}
                                    products_info = {loc: prods for loc, prods in
                                                     st.session_state.indoor_map.products.items()} if hasattr(
                                        st.session_state.indoor_map, 'products') else {}
                                    response = st.session_state.navigation_chain.invoke({
                                        "locations": str(locations_info),
                                        "current_location": st.session_state.current_location,
                                        "products_info": str(products_info),
                                        "question": nav_query
                                    })
                                    st.write("**Directions:**")
                                    st.write(response)
                else:
                    st.warning("Product not found in current store inventory")

        nav_col1, nav_col2 = st.columns([1, 1])

        with nav_col1:
            st.subheader("Navigation Controls")
            location_names = list(st.session_state.indoor_map.locations.keys())
            destination = st.selectbox("Where would you like to go?", ["Select destination"] + location_names)

            if st.session_state.current_location and destination != "Select destination":
                if st.button("Get Directions", key="main_directions"):
                    path = st.session_state.indoor_map.find_path(st.session_state.current_location, destination)
                    distance = st.session_state.indoor_map.calculate_distance(st.session_state.current_location,
                                                                              destination)
                    st.success(f"Route: {st.session_state.current_location} → {destination}")
                    st.info(f"Distance: ~{distance:.0f} units")

                    if hasattr(st.session_state.indoor_map,
                               'products') and destination in st.session_state.indoor_map.products:
                        products_at_dest = st.session_state.indoor_map.products[destination]
                        if products_at_dest:
                            st.write(f"**Products at {destination}:**")
                            for prod in products_at_dest[:3]:
                                st.write(f"• {prod['name']} - ₹{prod['price']}")

                    nav_query = f"Give me detailed directions from {st.session_state.current_location} to {destination}"
                    locations_info = {name: info['description'] for name, info in
                                      st.session_state.indoor_map.locations.items()}
                    products_info = {loc: prods for loc, prods in
                                     st.session_state.indoor_map.products.items()} if hasattr(
                        st.session_state.indoor_map, 'products') else {}
                    response = st.session_state.navigation_chain.invoke({
                        "locations": str(locations_info),
                        "current_location": st.session_state.current_location,
                        "products_info": str(products_info),
                        "question": nav_query
                    })
                    st.write("**Step-by-step Directions:**")
                    st.write(response)

            st.subheader("Smart Shopping List")
            shopping_items = st.text_area("Enter items (one per line):",
                                          placeholder="Basmati Rice\nTomatoes\nMilk\nBread")

            if shopping_items and st.button("Plan Shopping Route"):
                items_list = [item.strip() for item in shopping_items.split('\n') if item.strip()]
                shopping_route = []
                for item in items_list:
                    location = st.session_state.indoor_map.find_product_location(item)
                    if location:
                        shopping_route.append({'item': item, 'location': location})
                    else:
                        st.warning(f"{item} not found in store")

                if shopping_route:
                    st.success(f"Found locations for {len(shopping_route)} items")
                    unique_locations = []
                    for route_item in shopping_route:
                        if route_item['location'] not in unique_locations:
                            unique_locations.append(route_item['location'])

                    st.write("**Optimized Shopping Route:**")
                    current_loc = st.session_state.current_location
                    total_distance = 0
                    for i, location in enumerate(unique_locations):
                        distance = st.session_state.indoor_map.calculate_distance(current_loc, location)
                        total_distance += distance
                        st.write(f"{i + 1}. **{location}** (~{distance:.0f} units from previous)")
                        items_here = [r['item'] for r in shopping_route if r['location'] == location]
                        st.write(f"   Pick up: {', '.join(items_here)}")
                        current_loc = location
                    st.info(f"Total estimated distance: ~{total_distance:.0f} units")

        with nav_col2:
            st.subheader("Store Map")
            map_image = st.session_state.indoor_map.generate_visual_map(
                current_location=st.session_state.current_location,
                destination=destination if destination != "Select destination" else None
            )
            st.image(map_image, caption="Store Layout - Green: You are here, Red: Destination")
            st.markdown("""
            **Map Legend:**
            - Green: Your current location (QR anchor, auto-detect, or manual)
            - Red: Selected destination  
            - Blue: Other store sections & anchors  
            - Gray: Walls/Obstacles
            - Products: Check each section for available items
            """)
            if st.session_state.current_location in st.session_state.indoor_map.locations:
                current_info = st.session_state.indoor_map.locations[st.session_state.current_location]
                st.info(f"**Current Location:** {st.session_state.current_location}")
                st.write(current_info.get('description', ''))
                if hasattr(st.session_state.indoor_map, 'products'):
                    products_here = st.session_state.indoor_map.get_location_products(st.session_state.current_location)
                    if products_here:
                        st.write("**Products nearby:**")
                        for prod in products_here[:3]:
                            st.write(f"• {prod['name']} - ₹{prod['price']}")
                        if len(products_here) > 3:
                            st.caption(f"... and {len(products_here) - 3} more items")

# ==================== UNIVERSAL QUERY INTERFACE ====================

st.header("Ask Questions")

available_features = []
if st.session_state.chain is not None:
    available_features.append("Document Analysis")
if st.session_state.navigation_chain is not None:
    available_features.append("Indoor Navigation")

if not available_features:
    st.warning("Please set up at least one feature (upload a document or initialize navigation) to ask questions.")
else:
    if len(available_features) == 2:
        query_type = st.radio("What would you like to ask about?", available_features)
    else:
        query_type = available_features[0]

    input_method = st.radio("Select Input Method:", ("Text Input", "Voice Input"))
    query = ""

    if input_method == "Voice Input":
        if st.button("Record Voice"):
            audio_filename = record_audio()
            if audio_filename:
                lang_code = "en"
                if language == "Hindi":
                    lang_code = "hi-IN"
                elif language == "Odia":
                    lang_code = "or-IN"
                elif language == "Bengali":
                    lang_code = "bn-IN"
                elif language == "Tamil":
                    lang_code = "ta-IN"
                transcribed_text = transcribe_audio(audio_filename, lang=lang_code)
                if transcribed_text:
                    st.session_state.transcribed_query = transcribed_text
                    st.write(f"**Transcribed Query:** {transcribed_text}")

    if input_method == "Voice Input" and "transcribed_query" in st.session_state:
        query = st.session_state.transcribed_query
    else:
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

    output_method = st.radio("Select Output Method:", ("Text Only", "Text and Voice"))

    if st.button("Get Answer"):
        if query.strip() == "":
            st.error("Please enter or record a query before fetching an answer.")
        else:
            src_lang = "en"
            if language == "Hindi":
                src_lang = "hi"
            elif language == "Odia":
                src_lang = "or"
            elif language == "Bengali":
                src_lang = "bn"
            elif language == "Tamil":
                src_lang = "ta"

            if language != "English":
                translated_query, translation_success = safe_translate(query, src_lang, "en")
                if not translation_success:
                    st.warning("Query translation was not successful. Results might be less accurate.")
            else:
                translated_query = query
                translation_success = True

            if language != "English":
                st.write(f"**Translated Query (English):** {translated_query}")

            with st.spinner("Generating answer..."):
                if query_type == "Document Analysis":
                    if st.session_state.chain:
                        result = st.session_state.chain.invoke({"question": translated_query})
                    else:
                        result = "Document analysis chain not available. Please upload a document first."
                else:
                    if st.session_state.navigation_chain:
                        locations_info = {name: info['description'] for name, info in
                                          st.session_state.indoor_map.locations.items()}
                        products_info = {loc: prods for loc, prods in
                                         st.session_state.indoor_map.products.items()} if hasattr(
                            st.session_state.indoor_map, 'products') else {}
                        result = st.session_state.navigation_chain.invoke({
                            "locations": str(locations_info),
                            "current_location": st.session_state.current_location or "Unknown",
                            "products_info": str(products_info),
                            "question": translated_query
                        })
                    else:
                        result = "Navigation chain not available. Please initialize navigation first."

            if language != "English":
                st.write(f"**Raw Result (English):** {result}")

            if language != "English":
                translated_result, translation_success = safe_translate(result, "en", src_lang)
                if not translation_success:
                    st.warning("Response translation was not successful. Showing English response.")
                    final_result = result
                else:
                    final_result = translated_result
                    st.write(f"**Translated Result ({language}):** {translated_result}")
            else:
                final_result = result

            st.write("### Answer:")
            st.write(final_result)

            if output_method == "Text and Voice":
                tts_lang_code = get_tts_language_code(language)
                with st.spinner("Generating voice output..."):
                    text_for_speech = final_result
                    audio_bytes, tts_success = text_to_speech(text_for_speech, lang_code=tts_lang_code)
                    if audio_bytes and tts_success:
                        st.audio(audio_bytes, format="audio/mp3")
                        b64 = base64.b64encode(audio_bytes).decode()
                        href = f'<a href="data:audio/mp3;base64,{b64}" download="answer.mp3">Download audio response</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    elif not tts_success:
                        st.error("Could not generate voice output. Please try again or use text-only mode.")

# ==================== SIDEBAR ====================

offline_mode = st.sidebar.checkbox("Offline Mode (Skip translations)")
if offline_mode:
    st.sidebar.info("Running in offline mode. No translation services will be used.")

if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.sidebar.markdown("### Navigation Help")
    st.sidebar.markdown("""
    **Smart Shopping Features:**
    - **Product Search**: Find any item instantly
    - **QR Anchors (A1–A6)**: Scan to fix your position
    - **Auto Location**: GPS-like detection (defaults to Entrance)
    - **Smart Routing**: Optimized shopping paths
    - **Price Display**: See costs while navigating
    - **Shopping Lists**: Plan your entire trip

    **Try asking:**
    - "Where can I find Basmati Rice?"
    - "Show me all dairy products"
    - "What's the cheapest cooking oil?"
    - "Plan route for Rice, Milk, and Tomatoes"
    - "How much does Chicken cost?"
    """)

    if hasattr(st.session_state, 'indoor_map') and st.session_state.indoor_map and hasattr(st.session_state.indoor_map,
                                                                                           'products'):
        total_products = sum(len(products) for products in st.session_state.indoor_map.products.values())
        st.sidebar.metric("Products Available", total_products)
        st.sidebar.metric("Store Sections", len(st.session_state.indoor_map.locations))

st.sidebar.markdown("### Features Available")
feature_status = {
    "Document Analysis": st.session_state.chain is not None,
    "Indoor Navigation": st.session_state.navigation_chain is not None
}
for feature, status in feature_status.items():
    if status:
        st.sidebar.success(f"✅ {feature}")
    else:
        st.sidebar.info(f"❌ {feature}")

if st.session_state.current_location:
    st.sidebar.markdown("### Current Location")
    st.sidebar.success(f"You are at: **{st.session_state.current_location}**")
else:
    st.sidebar.markdown("### Location Status")
    st.sidebar.warning("Location not set (scan any A1–A6 QR, use auto-detect, or manual fallback)")

if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.sidebar.markdown("### Store Options")
    if st.sidebar.button("Reset Store Layout"):
        st.session_state.indoor_map = None
        st.session_state.navigation_chain = None
        st.session_state.current_location = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Usage Tips")
st.sidebar.markdown("""
**For best results:**
- Upload CSV with columns: Product Name, Category, Price, Veg/Non-Veg
- Upload PDF with store maps or information
- Use voice input for hands-free operation
- Enable location services for accurate positioning
- Scan QR anchors (A1-A6) for precise positioning
- Ask specific questions about products or directions
""")

if st.session_state.indoor_map and hasattr(st.session_state.indoor_map, 'products'):
    if st.sidebar.button("Export Store Data"):
        store_data = {
            'locations': st.session_state.indoor_map.locations,
            'products': st.session_state.indoor_map.products,
            'current_location': st.session_state.current_location
        }

        st.sidebar.download_button(
            label="Download Store Layout (JSON)",
            data=json.dumps(store_data, indent=2),
            file_name="store_layout.json",
            mime="application/json"
        )
