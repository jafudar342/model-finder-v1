# 🪐 ModelFinder

**ModelFinder** is a centralized discovery and community review platform for open-weight scientific machine learning models. It bridges the gap between Hugging Face model weights and GitHub research code, allowing researchers to search, evaluate, and review AI models tailored for physics, biology, chemistry, and more.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)
![Turso](https://img.shields.io/badge/Database-Turso_Cloud-black)
![Gemini](https://img.shields.io/badge/AI_Agent-Google_Gemini-orange)

## ✨ Core Features

* **Unified Scientific Search:** Queries both Hugging Face and GitHub concurrently to aggregate the best open-source models and research code.
* **Antigravity AI Agent:** Powered by Google Gemini 2.5 Flash, this built-in agent automatically scrapes GitHub READMEs to classify a repository's scientific domain, ML framework (PyTorch/JAX), hardware requirements, and provides a concise scientific summary.
* **Persistent Community Reviews:** Researchers can leave ratings and feedback on specific models. Reviews are stored permanently in the cloud.
* **Decoupled Serverless Database:** Uses Turso (libSQL) via a pure-Python HTTP REST API, ensuring zero data loss even on ephemeral cloud containers.
* **Lightweight UI:** A fast, responsive, single-page application built with Vanilla JS and Tailwind CSS.

## 🛠️ Tech Stack

* **Frontend:** HTML5, Vanilla JavaScript, Tailwind CSS (via CDN)
* **Backend:** Python, FastAPI, Uvicorn, Requests
* **Database:** Turso Cloud (SQLite edge database)
* **AI Engine:** Google GenAI (Gemini)
* **Deployment:** Render (Backend) + Custom DNS (GoDaddy)

## 🚀 Local Setup & Installation

Follow these steps to run ModelFinder locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/model-finder-v1.git
cd model-finder-v1
```

### 2. Set Up the Environment
Ensure you have Python installed (3.10+ recommended), then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
ModelFinder requires API keys to connect to Turso and Gemini. You can export these in your terminal or create a `.env` file (if you add `python-dotenv`):

```bash
export GEMINI_API_KEY="your_google_gemini_api_key"
export TURSO_DATABASE_URL="libsql://your-database-url.turso.io"
export TURSO_AUTH_TOKEN="your_turso_jwt_token"
```
*(Note: If Turso credentials are not provided, the app will gracefully fall back to a local `reviews.db` SQLite file).*

### 4. Run the Server
Start the FastAPI server using Uvicorn:
```bash
python server.py
# OR
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` in your browser to explore the platform.

## 📁 Project Structure

```text
├── index.html           # Main frontend UI (Tailwind + Vanilla JS)
├── server.py            # FastAPI backend, routing, and AI/Database logic
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignored files (API keys, local DBs, cache)
└── README.md            # Project documentation
```

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).