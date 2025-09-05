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
from PIL import Image, ImageDraw, ImageFont
import requests
from deep_translator import GoogleTranslator
import pandas as pd
import pyaudio
import wave
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import time
import json
import numpy as np
import math
import random

# Streamlit App Configuration
st.set_page_config(page_title="CartMapper", layout="wide")
st.title("CartMapper - Document Analysis & Indoor Navigation")

# Initialize session states
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

# Sidebar for main feature selection
feature_mode = st.sidebar.selectbox(
    "Select Feature:",
    ["Document Analysis", "Indoor Navigation", "Both Features"]
)

# Language selection (applies to both features)
language = st.radio("Select Language:", ("English", "Hindi", "Odia", "Bengali", "Tamil"))


# ==================== UTILITY FUNCTIONS ====================

# Initialize translator with retry mechanism
def get_translator(source='auto', target='en', max_retries=3):
    """Initialize translator with retry mechanism"""
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


# Initialize translator
translator = get_translator(source='auto', target='en')


def safe_translate(text, source_lang, target_lang, max_retries=2):
    """Safely translates text with retries and returns original text if translation fails."""
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
                    st.warning(f"Translation returned empty result. Using original text.")
                    return text, False
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Translation failed: {e}. Using original text.")
                return text, False
            else:
                st.info(f"Translation attempt {attempt + 1} failed. Retrying...")
                time.sleep(1)

    return text, False


# ==================== DOCUMENT PROCESSING FUNCTIONS ====================

def download_pdf_from_url(url):
    """Download a PDF from a URL and return its content as bytes."""
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
    """Process a PDF file and return a list of documents."""
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
    """Process a CSV file and return a list of documents."""
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
    """Set up the RAG (Retrieval-Augmented Generation) chain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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


# ==================== AUDIO PROCESSING FUNCTIONS ====================

def record_audio(filename="voice_input.wav", record_seconds=5, sample_rate=44100, chunk=1024):
    """Records audio from the microphone and saves it as a WAV file."""
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

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


def transcribe_audio(filename, lang="en"):
    """Converts speech from the audio file to text in the selected language."""
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


def text_to_speech(text, lang_code="en", max_retries=2):
    """Converts text to speech in the specified language with fallback to English."""
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
    """Maps the selected language to its code for text-to-speech."""
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Odia": "or",
        "Bengali": "bn",
        "Tamil": "ta"
    }
    return language_codes.get(language, "en")


# ==================== INDOOR NAVIGATION CLASSES ====================

class IndoorMap:
    """Class to handle indoor map operations"""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.locations = {}
        self.obstacles = []
        self.paths = []
        self.products = {}

    def add_location(self, name, x, y, description=""):
        """Add a location to the map"""
        self.locations[name] = {
            'x': x, 'y': y,
            'description': description
        }

    def add_obstacle(self, x1, y1, x2, y2):
        """Add a rectangular obstacle"""
        self.obstacles.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    def calculate_distance(self, loc1, loc2):
        """Calculate Euclidean distance between two locations"""
        if loc1 in self.locations and loc2 in self.locations:
            x1, y1 = self.locations[loc1]['x'], self.locations[loc1]['y']
            x2, y2 = self.locations[loc2]['x'], self.locations[loc2]['y']
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return float('inf')

    def find_path(self, start, end):
        """Simple pathfinding algorithm"""
        if start not in self.locations or end not in self.locations:
            return None

        start_pos = self.locations[start]
        end_pos = self.locations[end]

        # Simple direct path (can be enhanced with A* algorithm)
        path = [
            {'x': start_pos['x'], 'y': start_pos['y'], 'location': start},
            {'x': end_pos['x'], 'y': end_pos['y'], 'location': end}
        ]

        return path

    def find_product_location(self, product_name):
        """Find which location contains a specific product"""
        product_name_lower = product_name.lower()

        for location, products in self.products.items():
            for product in products:
                if product_name_lower in product['name'].lower():
                    return location
        return None

    def get_location_products(self, location_name):
        """Get all products at a specific location"""
        return self.products.get(location_name, [])

    def get_nearest_locations(self, current_location, max_distance=100):
        """Get locations within a certain distance"""
        if current_location not in self.locations:
            return []

        current_pos = self.locations[current_location]
        nearby = []

        for name, pos in self.locations.items():
            if name != current_location:
                distance = math.sqrt(
                    (pos['x'] - current_pos['x']) ** 2 +
                    (pos['y'] - current_pos['y']) ** 2
                )
                if distance <= max_distance:
                    nearby.append({
                        'name': name,
                        'distance': distance,
                        'description': pos['description']
                    })

        return sorted(nearby, key=lambda x: x['distance'])

    def get_products_by_type(self, product_type):
        """Filter products by type (Veg/Non-Veg)"""
        filtered_products = []

        for location, products in self.products.items():
            for product in products:
                if product.get('type', '').lower() == product_type.lower():
                    filtered_products.append({
                        'product': product,
                        'location': location
                    })

        return filtered_products

    def get_products_by_price_range(self, min_price, max_price):
        """Filter products by price range"""
        filtered_products = []

        for location, products in self.products.items():
            for product in products:
                price = product.get('price', 0)
                try:
                    price = float(price)
                    if min_price <= price <= max_price:
                        filtered_products.append({
                            'product': product,
                            'location': location
                        })
                except (ValueError, TypeError):
                    continue

        return sorted(filtered_products, key=lambda x: x['product'].get('price', 0))

    def search_products(self, search_term):
        """Search products by name or category"""
        search_term = search_term.lower()
        results = []

        for location, products in self.products.items():
            for product in products:
                product_name = product.get('name', '').lower()
                product_category = product.get('category', '').lower()

                if (search_term in product_name or
                        search_term in product_category):
                    results.append({
                        'product': product,
                        'location': location,
                        'match_score': (
                            2 if search_term in product_name else 1
                        )
                    })

        # Sort by match score and then by product name
        return sorted(results, key=lambda x: (-x['match_score'], x['product']['name']))

    def generate_visual_map(self, current_location=None, destination=None, path=None):
        """Generate a visual representation of the map"""
        img = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw obstacles
        for obstacle in self.obstacles:
            draw.rectangle([obstacle['x1'], obstacle['y1'], obstacle['x2'], obstacle['y2']],
                           fill='gray', outline='black')

        # Draw locations
        for name, loc in self.locations.items():
            color = 'blue'
            if name == current_location:
                color = 'green'
            elif name == destination:
                color = 'red'

            # Draw location as circle
            radius = 15
            draw.ellipse([loc['x'] - radius, loc['y'] - radius,
                          loc['x'] + radius, loc['y'] + radius],
                         fill=color, outline='black')

            # Draw location name
            try:
                # Try to use a font, fallback to default if not available
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            draw.text((loc['x'] + 20, loc['y']), name, fill='black', font=font)

        # Draw path
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                draw.line([path[i]['x'], path[i]['y'], path[i + 1]['x'], path[i + 1]['y']],
                          fill='red', width=3)

        return img


# ==================== NAVIGATION SETUP FUNCTIONS ====================

def get_user_location():
    """Simulate GPS-like location detection for indoor navigation"""
    # Simulate location detection delay
    time.sleep(1)

    # Simulate user being inside the store with some randomness
    inside_probability = 0.85  # 85% chance of being detected inside
    is_inside = random.random() < inside_probability

    # Simulate GPS accuracy
    accuracy = random.randint(3, 15)  # 3-15 meters accuracy

    # Additional location metadata
    location_data = {
        'inside_store': is_inside,
        'accuracy': accuracy,
        'timestamp': time.time(),
        'confidence': random.uniform(0.7, 0.95),  # Confidence level
        'detected_floor': 'Ground Floor' if is_inside else 'Unknown',
        'signal_strength': random.uniform(0.6, 1.0) if is_inside else random.uniform(0.1, 0.4)
    }

    return location_data


def create_dynamic_store_map(store_df):
    """Create a dynamic store map based on CSV data"""
    mall_map = IndoorMap(800, 600)

    # Add basic store locations
    mall_map.add_location("Entrance", 400, 550, "Main entrance to the store")

    # Get unique categories from the CSV
    if 'Category' in store_df.columns:
        categories = store_df['Category'].unique()

        # Define positions for different categories
        positions = [
            (150, 150), (650, 150), (150, 300), (650, 300),
            (150, 450), (650, 450), (400, 200), (400, 350)
        ]

        # Create locations for each category
        for i, category in enumerate(categories):
            if i < len(positions):
                x, y = positions[i]
                mall_map.add_location(
                    f"{category} Section",
                    x, y,
                    f"Section for {category} products"
                )
    else:
        # Default sections if no category column
        default_sections = [
            ("Grocery Section", 200, 200, "Fresh groceries and daily needs"),
            ("Dairy Section", 600, 200, "Milk, cheese, and dairy products"),
            ("Meat Section", 200, 400, "Fresh meat and poultry"),
            ("Vegetables Section", 600, 400, "Fresh vegetables and fruits"),
            ("Bakery Section", 400, 300, "Fresh bread and baked goods")
        ]

        for name, x, y, desc in default_sections:
            mall_map.add_location(name, x, y, desc)

    # Add utility locations
    mall_map.add_location("Customer Service", 100, 100, "Help and information desk")
    mall_map.add_location("Checkout", 400, 500, "Billing and payment counter")
    mall_map.add_location("Exit", 400, 50, "Store exit")

    # Add some obstacles (walls, pillars)
    mall_map.add_obstacle(350, 250, 450, 350)  # Central pillar
    mall_map.add_obstacle(0, 0, 800, 30)  # Top wall
    mall_map.add_obstacle(0, 0, 30, 600)  # Left wall
    mall_map.add_obstacle(770, 0, 800, 600)  # Right wall
    mall_map.add_obstacle(0, 570, 800, 600)  # Bottom wall

    # Store product information
    mall_map.products = {}

    # Group products by category and assign to locations
    if 'Category' in store_df.columns:
        for category in store_df['Category'].unique():
            category_products = store_df[store_df['Category'] == category]
            section_name = f"{category} Section"

            if section_name in mall_map.locations:
                mall_map.products[section_name] = []

                for _, row in category_products.iterrows():
                    product = {
                        'name': row.get('Product Name', row.get('Name', 'Unknown Product')),
                        'category': category,
                        'price': row.get('Price', row.get('Price (Rs)', 0)),
                        'type': row.get('Veg/Non-Veg', 'Unknown')
                    }
                    mall_map.products[section_name].append(product)
    else:
        # If no category column, distribute products across default sections
        sections = list(mall_map.locations.keys())
        product_sections = [s for s in sections if 'Section' in s]

        for section in product_sections:
            mall_map.products[section] = []

        for i, (_, row) in enumerate(store_df.iterrows()):
            if product_sections:  # Make sure we have sections
                section_idx = i % len(product_sections)
                section_name = product_sections[section_idx]

                product = {
                    'name': row.get('Product Name', row.get('Name', f'Product {i + 1}')),
                    'category': row.get('Category', 'General'),
                    'price': row.get('Price', row.get('Price (Rs)', 0)),
                    'type': row.get('Veg/Non-Veg', 'Unknown')
                }
                mall_map.products[section_name].append(product)

    return mall_map


def create_sample_mall_map():
    """Create a sample shopping mall map"""
    mall_map = IndoorMap(800, 600)

    # Add main areas
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

    # Add some obstacles (walls, pillars)
    mall_map.add_obstacle(350, 250, 450, 350)  # Central pillar
    mall_map.add_obstacle(0, 0, 800, 50)  # Top wall
    mall_map.add_obstacle(0, 0, 50, 600)  # Left wall
    mall_map.add_obstacle(750, 0, 800, 600)  # Right wall
    mall_map.add_obstacle(0, 550, 800, 600)  # Bottom wall

    # Add sample products
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


def setup_navigation_chain():
    """Set up the navigation-specific LLM chain"""
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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

    chain = (
            prompt
            | llm
            | StrOutputParser()
    )

    return chain


# ==================== UI RENDERING BASED ON FEATURE MODE ====================

if feature_mode in ["Document Analysis", "Both Features"]:
    st.header("Document Analysis")

    # File uploader for PDF and CSV
    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

    # Option to scan QR code from camera or upload an image
    qr_option = st.radio("QR Code Input Method:", ("Upload QR Image", "Scan QR from Camera"))

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
        if qr_option == "Upload QR Image" and uploaded_qr:
            st.error("No QR code found in the uploaded image.")
        elif qr_option == "Scan QR from Camera" and camera_image:
            st.error("No QR code found in the camera image.")

    # Process uploaded file (PDF or CSV)
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

if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.header("Indoor Navigation & Product Finder")

    # CSV upload for store layout
    store_csv = st.file_uploader("Upload Store Inventory (CSV)", type=["csv"], key="store_csv")

    # Initialize navigation system
    if store_csv is not None:
        # Load store data from CSV
        store_df = pd.read_csv(store_csv)
        st.session_state.indoor_map = create_dynamic_store_map(store_df)
        st.session_state.navigation_chain = setup_navigation_chain()

        # Display store info
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
            st.session_state.navigation_chain = setup_navigation_chain()

    if st.session_state.indoor_map is not None:
        # Simulate location detection (like Google Maps)
        location_col1, location_col2 = st.columns([2, 1])

        with location_col2:
            st.subheader("Location Services")

            # Simulate GPS-like location detection
            if st.button("Detect My Location"):
                user_location = get_user_location()
                if user_location['inside_store']:
                    st.success("Location detected: Inside store")
                    st.info(f"Accuracy: ±{user_location['accuracy']} meters")
                    # Auto-set to entrance if no current location
                    if not st.session_state.current_location:
                        st.session_state.current_location = "Entrance"
                        st.success("Set current location to: Entrance")
                else:
                    st.warning("You appear to be outside the store")

            # Manual location override
            location_names = list(st.session_state.indoor_map.locations.keys())
            current_location = st.selectbox(
                "Or manually select location:",
                ["Auto-detect (Entrance)"] + location_names,
                index=0 if st.session_state.current_location is None else
                location_names.index(
                    st.session_state.current_location) + 1 if st.session_state.current_location in location_names else 0
            )

            if current_location == "Auto-detect (Entrance)":
                st.session_state.current_location = "Entrance"
            else:
                st.session_state.current_location = current_location

            st.success(f"Current: **{st.session_state.current_location}**")

        with location_col1:
            # Product search interface
            st.subheader("Product Search")

            search_query = st.text_input("Search for products:", placeholder="e.g., Basmati Rice, Tomato, Chicken")

            if search_query:
                # Search in product database
                found_products = []
                locations_with_products = []

                if hasattr(st.session_state.indoor_map, 'products'):
                    for location, products in st.session_state.indoor_map.products.items():
                        for product in products:
                            if search_query.lower() in product['name'].lower():
                                found_products.append({
                                    'product': product,
                                    'location': location
                                })
                                if location not in locations_with_products:
                                    locations_with_products.append(location)

                if found_products:
                    st.success(f"Found {len(found_products)} matching product(s)")

                    for item in found_products[:5]:  # Show top 5 matches
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

                                    # Generate directions using LLM
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

        # Main navigation interface
        nav_col1, nav_col2 = st.columns([1, 1])

        with nav_col1:
            st.subheader("Navigation Controls")

            # Destination selection
            location_names = list(st.session_state.indoor_map.locations.keys())
            destination = st.selectbox(
                "Where would you like to go?",
                ["Select destination"] + location_names
            )

            # Quick navigation buttons
            if st.session_state.current_location and destination != "Select destination":
                if st.button("Get Directions", key="main_directions"):
                    path = st.session_state.indoor_map.find_path(st.session_state.current_location, destination)
                    distance = st.session_state.indoor_map.calculate_distance(st.session_state.current_location,
                                                                              destination)

                    st.success(f"Route: {st.session_state.current_location} → {destination}")
                    st.info(f"Distance: ~{distance:.0f} units")

                    # Show products at destination
                    if hasattr(st.session_state.indoor_map,
                               'products') and destination in st.session_state.indoor_map.products:
                        products_at_dest = st.session_state.indoor_map.products[destination]
                        if products_at_dest:
                            st.write(f"**Products at {destination}:**")
                            for prod in products_at_dest[:3]:  # Show first 3
                                st.write(f"• {prod['name']} - ₹{prod['price']}")

                    # Generate directions using enhanced LLM
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

            # Shopping list feature
            st.subheader("Smart Shopping List")
            shopping_items = st.text_area("Enter items (one per line):",
                                          placeholder="Basmati Rice\nTomatoes\nMilk\nBread")

            if shopping_items and st.button("Plan Shopping Route"):
                items_list = [item.strip() for item in shopping_items.split('\n') if item.strip()]

                # Find locations for each item
                shopping_route = []
                for item in items_list:
                    location = st.session_state.indoor_map.find_product_location(item)
                    if location:
                        shopping_route.append({'item': item, 'location': location})
                    else:
                        st.warning(f"{item} not found in store")

                if shopping_route:
                    st.success(f"Found locations for {len(shopping_route)} items")

                    # Optimize route (simple: unique locations in order)
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

                        # Show items at this location
                        items_here = [r['item'] for r in shopping_route if r['location'] == location]
                        st.write(f"   Pick up: {', '.join(items_here)}")

                        current_loc = location

                    st.info(f"Total estimated distance: ~{total_distance:.0f} units")

        with nav_col2:
            st.subheader("Store Map")

            # Generate and display map
            map_image = st.session_state.indoor_map.generate_visual_map(
                current_location=st.session_state.current_location,
                destination=destination if destination != "Select destination" else None
            )

            st.image(map_image, caption="Store Layout - Green: You are here, Red: Destination")

            # Enhanced map legend
            st.markdown("""
            **Map Legend:**
            - Green: Your current location
            - Red: Selected destination  
            - Blue: Other store sections
            - Gray: Walls/Obstacles
            - Products: Check each section for available items
            """)

            # Quick location info
            if st.session_state.current_location in st.session_state.indoor_map.locations:
                current_info = st.session_state.indoor_map.locations[st.session_state.current_location]
                st.info(f"**Current Location:** {st.session_state.current_location}")
                st.write(current_info.get('description', ''))

                # Show products at current location
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

# Determine which chains are available
available_features = []
if st.session_state.chain is not None:
    available_features.append("Document Analysis")
if st.session_state.navigation_chain is not None:
    available_features.append("Indoor Navigation")

if not available_features:
    st.warning("Please set up at least one feature (upload a document or initialize navigation) to ask questions.")
else:
    # Query type selection (only show if both features are available)
    if len(available_features) == 2:
        query_type = st.radio("What would you like to ask about?", available_features)
    else:
        query_type = available_features[0]

    # Input method selection
    input_method = st.radio("Select Input Method:", ("Text Input", "Voice Input"))

    query = ""

    if input_method == "Voice Input":
        if st.button("Record Voice"):
            audio_filename = record_audio()
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

    # Use transcribed query if voice input was used
    if input_method == "Voice Input" and "transcribed_query" in st.session_state:
        query = st.session_state.transcribed_query
    else:
        # Text input with language-specific prompts
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

    # Output method selection
    output_method = st.radio("Select Output Method:", ("Text Only", "Text and Voice"))

    # Process query when "Get Answer" is clicked
    if st.button("Get Answer"):
        if query.strip() == "":
            st.error("Please enter or record a query before fetching an answer.")
        else:
            # Determine source language code based on selected language
            src_lang = "en"
            if language == "Hindi":
                src_lang = "hi"
            elif language == "Odia":
                src_lang = "or"
            elif language == "Bengali":
                src_lang = "bn"
            elif language == "Tamil":
                src_lang = "ta"

            # Translate input to English if needed
            if language != "English":
                translated_query, translation_success = safe_translate(query, src_lang, "en")
                if not translation_success:
                    st.warning("Query translation was not successful. Results might be less accurate.")
            else:
                translated_query = query
                translation_success = True

            # Show translated query only if it was actually translated
            if language != "English":
                st.write(f"**Translated Query (English):** {translated_query}")

            # Invoke the appropriate chain based on query type
            with st.spinner("Generating answer..."):
                if query_type == "Document Analysis":
                    result = st.session_state.chain.invoke(translated_query)
                else:  # Indoor Navigation
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

            # Show raw result from chain
            if language != "English":
                st.write(f"**Raw Result (English):** {result}")

            # Translate output back to the selected language if needed
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

            # Display the final answer
            st.write("### Answer:")
            st.write(final_result)

            # Generate voice output if selected
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

# Add offline mode toggle in sidebar
offline_mode = st.sidebar.checkbox("Offline Mode (Skip translations)")
if offline_mode:
    st.sidebar.info("Running in offline mode. No translation services will be used.")

# Navigation help in sidebar
if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.sidebar.markdown("### Navigation Help")
    st.sidebar.markdown("""
    **Smart Shopping Features:**
    - **Product Search**: Find any item instantly
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

    # Store statistics if available
    if hasattr(st.session_state, 'indoor_map') and st.session_state.indoor_map and hasattr(st.session_state.indoor_map,
                                                                                           'products'):
        total_products = sum(len(products) for products in st.session_state.indoor_map.products.values())
        st.sidebar.metric("Products Available", total_products)
        st.sidebar.metric("Store Sections", len(st.session_state.indoor_map.locations))

# Feature information
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

# Current location display
if st.session_state.current_location:
    st.sidebar.markdown("### Current Location")
    st.sidebar.success(f"You are at: **{st.session_state.current_location}**")
else:
    st.sidebar.markdown("### Location Status")
    st.sidebar.warning("Location not set (will default to Entrance)")

# Add store switching capability
if feature_mode in ["Indoor Navigation", "Both Features"]:
    st.sidebar.markdown("### Store Options")
    if st.sidebar.button("Reset Store Layout"):
        st.session_state.indoor_map = None
        st.session_state.navigation_chain = None
        st.session_state.current_location = None
        st.rerun()
