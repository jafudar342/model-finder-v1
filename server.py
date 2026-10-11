import os
import re
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import HfApi

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

app = FastAPI(title="ModelFinder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 1. ENVIRONMENT & TURSO CLOUD DB CONFIGURATION
# ==========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()

# Fallback to local turso auth.txt if running locally without env vars
if (not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN) and os.path.exists("turso auth.txt"):
    try:
        with open("turso auth.txt", "r", encoding="utf-8") as f:
            content = f.read()
            url_match = re.search(r"(libsql://[^\s\"']+)", content)
            tok_match = re.search(r"(eyJ[a-zA-Z0-9_\-\.]+)", content)
            if url_match and not TURSO_DATABASE_URL:
                TURSO_DATABASE_URL = url_match.group(1).strip()
            if tok_match and not TURSO_AUTH_TOKEN:
                TURSO_AUTH_TOKEN = tok_match.group(1).strip()
    except Exception as e:
        print(f"[Database Warning] Could not parse local turso auth.txt: {e}")

gemini_client = None
if HAS_GENAI and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[Antigravity Warning] Failed to initialize Gemini client: {e}")

hf_api = HfApi()
LOCAL_DB_PATH = "reviews.db"


def get_turso_http_url() -> Optional[str]:
    if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
        return None
    base = TURSO_DATABASE_URL.replace("libsql://", "https://").rstrip("/")
    return f"{base}/v2/pipeline"


def db_execute(sql: str, args: list = None) -> list:
    """
    Executes a SQL query against Turso Cloud over HTTPS (v2/pipeline).
    Falls back to local SQLite if Turso credentials are not configured.
    Returns a list of dicts representing rows.
    """
    args = args or []
    turso_url = get_turso_http_url()

    if turso_url:
        formatted_args = []
        for a in args:
            if a is None:
                formatted_args.append({"type": "null"})
            elif isinstance(a, int):
                formatted_args.append({"type": "integer", "value": str(a)})
            elif isinstance(a, float):
                formatted_args.append({"type": "float", "value": a})
            else:
                formatted_args.append({"type": "text", "value": str(a)})

        payload = {
            "requests": [
                {"type": "execute", "stmt": {"sql": sql, "args": formatted_args}},
                {"type": "close"}
            ]
        }
        resp = requests.post(
            turso_url,
            headers={
                "Authorization": f"Bearer {TURSO_AUTH_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        first_res = data.get("results", [{}])[0]
        if first_res.get("type") == "error":
            raise RuntimeError(f"Turso SQL Error: {first_res.get('error')}")

        result_obj = first_res.get("response", {}).get("result", {})
        cols = [c.get("name") for c in result_obj.get("cols", [])]
        rows = []
        for raw_row in result_obj.get("rows", []):
            row_dict = {}
            for idx, cell in enumerate(raw_row):
                col_name = cols[idx] if idx < len(cols) else f"col_{idx}"
                val = cell.get("value")
                if cell.get("type") == "integer" and val is not None:
                    val = int(val)
                elif cell.get("type") == "float" and val is not None:
                    val = float(val)
                row_dict[col_name] = val
            rows.append(row_dict)
        return rows

    # Local SQLite Fallback
    with sqlite3.connect(LOCAL_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, args)
        conn.commit()
        if cur.description:
            return [dict(r) for r in cur.fetchall()]
        return []


def init_db():
    try:
        db_execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                username TEXT NOT NULL,
                user_role TEXT NOT NULL,
                rating INTEGER NOT NULL,
                date TEXT NOT NULL,
                comment TEXT NOT NULL,
                helpful_count INTEGER DEFAULT 0
            )
        """)
        mode = "Turso Cloud HTTP" if get_turso_http_url() else "Local SQLite"
        print(f"[Database] Initialized successfully ({mode}).")
    except Exception as e:
        print(f"[Database Error] Failed to initialize database: {e}")


init_db()


# ==========================================================
# 2. SCIENTIFIC DOMAIN CLASSIFICATION & GEMINI 3 FLASH
# ==========================================================
VALID_DOMAINS = {
    "Physics": ["Fluid Dynamics", "Astrophysics", "PINNs", "Quantum Mechanics", "Applied Physics"],
    "Chemistry": ["Molecular Modeling", "Drug Discovery", "Reaction Kinetics"],
    "Biology": ["Genomics", "Oncology / Breast Cancer", "Gut Microbiome", "Neuroscience", "Computational Biology"],
    "Earth & Climate": ["Atmospheric Modeling", "Oceanography", "Remote Sensing"],
    "Materials Science": ["Materials Science", "Solid State DFT", "Perovskite Modeling"],
}

DOMAIN_SEARCH_EXPANSIONS = {
    "earth & climate": "climate weather geospatial atmospheric earth remote sensing",
    "earth climate": "climate weather geospatial atmospheric earth remote sensing",
    "materials science": "materials crystal dft perovskite lattice molecular dynamics",
    "physics": "physics pinn fluid dynamics turbulence quantum neural operator",
    "chemistry": "chemistry molecular gnn drug discovery admet smiles",
    "biology": "biology genomics protein dna microbiome neuroscience medical",
}


def infer_domain_fallback(query: str, repo_name: str, desc: str, tags: list) -> tuple[str, str]:
    """
    Deterministic scientific domain classifier used when Gemini is unavailable
    or returns an unrecognized domain. Guarantees compatibility with UI filters.
    """
    q_lower = (query or "").lower()
    corpus = f"{q_lower} {repo_name} {desc} {' '.join(tags)}".lower()

    # If the user explicitly searched for one of the 5 main domains, prioritize it
    for dom in VALID_DOMAINS:
        if dom.lower() in q_lower or dom.lower().replace("&", "").strip() in q_lower:
            return dom, VALID_DOMAINS[dom][0]

    if any(k in corpus for k in ["climate", "earth", "weather", "ocean", "atmospher", "geospatial", "sentinel", "landsat", "flood", "meteorolog"]):
        if "ocean" in corpus:
            return "Earth & Climate", "Oceanography"
        if any(k in corpus for k in ["sentinel", "landsat", "satellite", "remote"]):
            return "Earth & Climate", "Remote Sensing"
        return "Earth & Climate", "Atmospheric Modeling"

    if any(k in corpus for k in ["material", "crystal", "dft", "perovskite", "lattice", "bandgap", "solid-state", "phonon"]):
        if "perovskite" in corpus:
            return "Materials Science", "Perovskite Modeling"
        return "Materials Science", "Solid State DFT"

    if any(k in corpus for k in ["chem", "molecule", "molecular", "drug", "admet", "smiles", "pharmac", "kinetics", "force-field"]):
        if any(k in corpus for k in ["drug", "admet", "pharmac"]):
            return "Chemistry", "Drug Discovery"
        return "Chemistry", "Molecular Modeling"

    if any(k in corpus for k in ["physic", "pinn", "fluid", "quantum", "navier", "turbulence", "astro", "pde", "hamiltonian"]):
        if "pinn" in corpus or "pde" in corpus:
            return "Physics", "PINNs"
        if any(k in corpus for k in ["fluid", "navier", "turbulence", "cfd"]):
            return "Physics", "Fluid Dynamics"
        if "quantum" in corpus:
            return "Physics", "Quantum Mechanics"
        if "astro" in corpus:
            return "Physics", "Astrophysics"
        return "Physics", "Applied Physics"

    if any(k in corpus for k in ["bio", "genom", "dna", "rna", "protein", "microbiome", "gut", "16s", "brain", "neuro", "mri", "tumor", "oncol", "cancer", "cell", "medical"]):
        if any(k in corpus for k in ["brain", "neuro", "cortex", "mri", "eeg"]):
            return "Biology", "Neuroscience"
        if any(k in corpus for k in ["microbiome", "gut", "16s", "metagenom", "asv"]):
            return "Biology", "Gut Microbiome"
        if any(k in corpus for k in ["tumor", "oncol", "cancer", "breast"]):
            return "Biology", "Oncology / Breast Cancer"
        if any(k in corpus for k in ["genom", "dna", "rna", "nucleotide", "chromatin"]):
            return "Biology", "Genomics"
        return "Biology", "Computational Biology"

    return "Physics", "Applied Physics"


def infer_framework(text: str, tags: list) -> str:
    corpus = f"{text} {' '.join(tags)}".lower()
    if "jax" in corpus or "flax" in corpus:
        return "JAX"
    if "tensorflow" in corpus or "keras" in corpus or "#tf" in corpus:
        return "TensorFlow"
    if "onnx" in corpus:
        return "ONNX"
    return "PyTorch"


def analyze_with_gemini(query: str, repo_name: str, raw_desc: str, tags: list) -> dict:
    fallback_dom, fallback_sub = infer_domain_fallback(query, repo_name, raw_desc, tags)
    fallback_fw = infer_framework(f"{repo_name} {raw_desc}", tags)
    default_result = {
        "primary_domain": fallback_dom,
        "sub_domain": fallback_sub,
        "framework": fallback_fw,
        "hardware_req": "16GB VRAM",
        "vram_gb": 16,
        "summary": raw_desc if raw_desc and len(raw_desc) > 15 else f"Open-weight scientific model/repository {repo_name} for {fallback_sub}."
    }

    if not gemini_client:
        return default_result

    prompt = (
        f"Analyze this scientific AI model/repository for search query '{query}'.\n"
        f"Name: {repo_name}\n"
        f"Description: {raw_desc}\n"
        f"Tags: {', '.join(tags[:8])}\n\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "- primary_domain: MUST be one of ['Physics', 'Chemistry', 'Biology', 'Earth & Climate', 'Materials Science']\n"
        "- sub_domain: concise scientific sub-discipline (e.g., 'Fluid Dynamics', 'Neuroscience', 'Atmospheric Modeling', 'Drug Discovery', 'Solid State DFT')\n"
        "- framework: one of ['PyTorch', 'JAX', 'TensorFlow', 'ONNX']\n"
        "- hardware_req: concise GPU spec (e.g., '12GB VRAM', '16GB VRAM', '24GB+ VRAM')\n"
        "- vram_gb: integer (8, 12, 16, or 24)\n"
        "- summary: 1-2 sentence technical description of what the model does."
    )

    for model_name in ("gemini-3-flash-preview", "gemini-3.0-flash"):
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = (response.text or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            parsed = json.loads(text)

            dom = parsed.get("primary_domain", fallback_dom)
            if dom not in VALID_DOMAINS:
                dom = fallback_dom

            return {
                "primary_domain": dom,
                "sub_domain": parsed.get("sub_domain") or fallback_sub,
                "framework": parsed.get("framework") if parsed.get("framework") in ["PyTorch", "JAX", "TensorFlow", "ONNX"] else fallback_fw,
                "hardware_req": parsed.get("hardware_req") or "16GB VRAM",
                "vram_gb": int(parsed.get("vram_gb") or 16),
                "summary": parsed.get("summary") or default_result["summary"],
            }
        except Exception as e:
            if "404" in str(e) and model_name == "gemini-3-flash-preview":
                continue
            print(f"[Antigravity Warning] Gemini note ({model_name}): {e}")
            break

    return default_result


def compute_health(engagement: int, updated_str: str) -> tuple[str, int]:
    score = 55
    if engagement >= 5000:
        score += 35
    elif engagement >= 500:
        score += 25
    elif engagement >= 50:
        score += 15
    elif engagement >= 10:
        score += 5

    if any(yr in str(updated_str) for yr in ["2026", "2025"]):
        score += 10

    score = min(98, max(25, score))
    if score >= 88:
        return "A+", score
    if score >= 75:
        return "A", score
    if score >= 60:
        return "B", score
    return "C", score


# ==========================================================
# 3. API ENDPOINTS
# ==========================================================
class ReviewCreate(BaseModel):
    username: str = "Verified Researcher"
    user_role: str = "AI Research Scientist"
    rating: int = 5
    comment: str


@app.get("/")
def serve_home():
    return FileResponse("index.html")


@app.get("/api/search")
def search_models(q: str = Query("scientific machine learning")):
    raw_query = (q or "scientific machine learning").strip()
    clean_query = re.sub(r"[&/\\#,+()$~%'\":*?<>{}]", " ", raw_query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()

    # Expand broad domain names into rich scientific keywords
    search_terms = DOMAIN_SEARCH_EXPANSIONS.get(raw_query.lower(), clean_query)
    hf_query = clean_query if raw_query.lower() not in DOMAIN_SEARCH_EXPANSIONS else search_terms.split()[0]

    # Fetch review counts from Turso DB
    review_counts = {}
    try:
        rows = db_execute("SELECT model_id, COUNT(*) as cnt FROM reviews GROUP BY model_id")
        for r in rows:
            review_counts[r["model_id"]] = int(r["cnt"])
    except Exception as e:
        print(f"[Database Warning] Could not fetch review counts: {e}")

    results = []

    # 1. Hugging Face Search (without deprecated 'direction' arg)
    try:
        hf_models = list(hf_api.list_models(
            search=hf_query,
            sort="downloads",
            limit=6
        ))
        for m in hf_models:
            model_id_raw = getattr(m, "id", "") or getattr(m, "modelId", "")
            if not model_id_raw:
                continue
            short_name = model_id_raw.split("/")[-1]
            downloads = int(getattr(m, "downloads", 0) or 0)
            raw_tags = [f"#{t}" for t in (getattr(m, "tags", []) or [])[:5]]
            last_mod = getattr(m, "lastModified", None)
            updated_str = str(last_mod)[:10] if last_mod else "2026-09-15"

            dom, sub = infer_domain_fallback(raw_query, short_name, model_id_raw, raw_tags)
            fw = infer_framework(model_id_raw, raw_tags)
            tier, score = compute_health(downloads, updated_str)
            internal_id = f"hf-{model_id_raw.replace('/', '-')}"

            results.append({
                "id": internal_id,
                "model_name": short_name,
                "source": "Hugging Face",
                "primary_domain": dom,
                "sub_domain": sub,
                "tags": raw_tags or ["#open-weights", "#scientific-ml"],
                "framework": fw,
                "hardware_req": "12GB VRAM",
                "vram_gb": 12,
                "description": f"Scientific open-weight model ({model_id_raw}) indexed from Hugging Face for {sub}.",
                "sample_stats": f"📥 {downloads:,} downloads",
                "sample_engagement_num": downloads,
                "last_updated": updated_str,
                "health_tier": tier,
                "health_score": score,
                "review_count": review_counts.get(internal_id, 0),
                "repo_url": f"https://huggingface.co/{model_id_raw}"
            })
    except Exception as e:
        print(f"[Error] Hugging Face search failed: {e}")

    # 2. GitHub Repositories Search
    try:
        gh_q = f"{search_terms} machine learning"
        gh_resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": gh_q, "sort": "stars", "order": "desc", "per_page": 6},
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "ModelFinder-App"},
            timeout=8
        )
        if gh_resp.ok:
            items = gh_resp.json().get("items", [])
            for idx, repo in enumerate(items):
                repo_id = f"gh-{repo.get('id')}"
                name = repo.get("name", "Repository")
                raw_desc = repo.get("description") or ""
                topics = [f"#{t}" for t in (repo.get("topics") or [])[:5]]
                stars = int(repo.get("stargazers_count", 0) or 0)
                updated_str = (repo.get("updated_at") or "2026-10-01")[:10]
                html_url = repo.get("html_url", "https://github.com")

                # Run Gemini enrichment on top 2 results for speed, deterministic classifier on rest
                if idx < 2 and gemini_client:
                    ai_meta = analyze_with_gemini(raw_query, name, raw_desc, topics)
                else:
                    dom, sub = infer_domain_fallback(raw_query, name, raw_desc, topics)
                    ai_meta = {
                        "primary_domain": dom,
                        "sub_domain": sub,
                        "framework": infer_framework(f"{name} {raw_desc}", topics),
                        "hardware_req": "16GB VRAM",
                        "vram_gb": 16,
                        "summary": raw_desc or f"Scientific machine learning repository {name} for {sub}."
                    }

                tier, score = compute_health(stars, updated_str)
                results.append({
                    "id": repo_id,
                    "model_name": name,
                    "source": "GitHub",
                    "primary_domain": ai_meta["primary_domain"],
                    "sub_domain": ai_meta["sub_domain"],
                    "tags": topics or ["#scientific-ml", "#open-source"],
                    "framework": ai_meta["framework"],
                    "hardware_req": ai_meta["hardware_req"],
                    "vram_gb": ai_meta["vram_gb"],
                    "description": ai_meta["summary"],
                    "sample_stats": f"⭐ {stars:,} stars",
                    "sample_engagement_num": stars,
                    "last_updated": updated_str,
                    "health_tier": tier,
                    "health_score": score,
                    "review_count": review_counts.get(repo_id, 0),
                    "repo_url": html_url
                })
    except Exception as e:
        print(f"[Error] GitHub search failed: {e}")

    return {"models": results}


@app.get("/api/reviews/{model_id}")
def get_reviews(model_id: str):
    try:
        rows = db_execute(
            "SELECT id, username, user_role, rating, date, comment, helpful_count FROM reviews WHERE model_id = ? ORDER BY rowid DESC",
            [model_id]
        )
        formatted = [
            {
                "id": r["id"],
                "username": r["username"],
                "userRole": r["user_role"],
                "rating": int(r["rating"]),
                "date": r["date"],
                "comment": r["comment"],
                "helpfulCount": int(r.get("helpful_count") or 0),
            }
            for r in rows
        ]
        return {"reviews": formatted}
    except Exception as e:
        print(f"[Database Error] Failed to fetch reviews for {model_id}: {e}")
        return {"reviews": []}


@app.post("/api/reviews/{model_id}")
def create_review(model_id: str, payload: ReviewCreate):
    review_id = f"rev-{uuid.uuid4().hex[:8]}"
    now_str = datetime.utcnow().strftime("%b %d, %Y")
    username = (payload.username or "Verified Researcher").strip()
    user_role = (payload.user_role or "AI Research Scientist").strip()
    rating = max(1, min(5, int(payload.rating)))
    comment = payload.comment.strip()

    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        db_execute(
            """
            INSERT INTO reviews (id, model_id, username, user_role, rating, date, comment, helpful_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            [review_id, model_id, username, user_role, rating, now_str, comment]
        )
    except Exception as e:
        print(f"[Database Error] Failed to insert review: {e}")
        raise HTTPException(status_code=500, detail="Failed to save review to database")

    return {
        "id": review_id,
        "username": username,
        "userRole": user_role,
        "rating": rating,
        "date": now_str,
        "comment": comment,
        "helpfulCount": 0
    }


@app.post("/api/reviews/{review_id}/helpful")
def mark_review_helpful(review_id: str):
    try:
        db_execute(
            "UPDATE reviews SET helpful_count = COALESCE(helpful_count, 0) + 1 WHERE id = ?",
            [review_id]
        )
        return {"status": "ok"}
    except Exception as e:
        print(f"[Database Error] Failed to increment helpful count: {e}")
        raise HTTPException(status_code=500, detail="Failed to update helpful count")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
