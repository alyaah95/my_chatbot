# Egyptia — Historical & Geographical RAG Platform

An interactive platform for exploring Egyptian history enriched with custom SVG maps and an AI-powered RAG chatbot (utilizing Groq API) with web search fallback.

---

## 🚀 Live Demo

- **Application URL:** [https://egyptia-two.vercel.app/](https://egyptia-two.vercel.app/)

---

## 💡 Note for Visitors & Evaluators
> ⚠️ **API Usage Notice:** The built-in AI chatbot operates on a free-tier API plan. Please be mindful and avoid sending an excessive number of consecutive or heavy queries in a short timeframe so the service remains available for everyone to test. Thank you for your understanding!

---

## 🛠️ Tech Stack

- **Frontend:** React.js, Vite, Custom SVG Maps, Tailwind CSS
- **Backend:** Python, Flask
- **AI & RAG:** Groq API, Hugging Face API
- **Asset Storage:** Cloudinary
- **Deployment:** Vercel

---

## ⚙️ Environment Variables Templates (`.env`)

> ⚠️ **Note:** Do NOT commit your real `.env` file to version control. Replace the placeholder values with your own keys.

### 1. Backend (`egypt-backend/.env`)
```env
GROQ_API_KEY=your_groq_api_key_here
HF_API_TOKEN=your_huggingface_api_token_here
```

### 2. Frontend (`egypt-frontend/.env`)
```env
# For local development:
VITE_API_BASE_URL=http://127.0.0.1:5000


---

## 💻 Local Setup & Development Instructions

To run **Egyptia** locally on your machine, follow these steps:

### Prerequisites
- Node.js & npm (for frontend)
- Python 3.8+ & pip (for backend)

---

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd my_chatbot
```

---

### Step 2: Set Up and Run the Flask Backend (`egypt-backend`)

1. Navigate to the backend directory:
   ```bash
   cd egypt-backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `egypt-backend` folder and add your `GROQ_API_KEY` and `HF_API_TOKEN`.
5. Run the Flask server:
   ```bash
   python app.py
   # or
   flask run
   ```
   The backend server will run on `http://127.0.0.1:5000`.

---

### Step 3: Set Up and Run the React Frontend (`egypt-frontend`)

1. Open a new terminal tab/window and navigate to the frontend directory:
   ```bash
   cd egypt-frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `egypt-frontend` folder and set the base URL to your local backend:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:5000
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```
5. Open your browser and navigate to the local URL provided by Vite (typically `http://localhost:5173`).
