import os
import json
import sqlite3
import ast
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
from bootstrap_flask import Bootstrap5
from dotenv import load_dotenv

# Gemini API (google-generativeai)
try:
    import google.generativeai as genai
except Exception:  # Keep import simple for beginners
    genai = None

# -----------------------------
# Setup and configuration
# -----------------------------
load_dotenv()  # Load variables from .env if present

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
Bootstrap5(app)

DATABASE_PATH = os.path.join(os.getcwd(), "pyquest.db")
SINGLE_USER_ID = 1  # No auth in MVP; single-user mode

# -----------------------------
# SQLite helpers (simple & beginner-friendly)
# -----------------------------

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    # Projects table: store generated ideas
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            steps TEXT NOT NULL,
            user_id INTEGER NOT NULL
        );
        """
    )
    # Submissions table: store code submissions and points
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            points INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()


# -----------------------------
# Gemini helpers
# -----------------------------

def configure_gemini_if_available() -> Optional["genai.GenerativeModel"]:
    """Create a Gemini client (gemini-1.5-flash) if API key and library are available.

    Returns a configured GenerativeModel or None if unavailable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception:
        # If configuration fails (e.g., invalid key), fall back to None
        return None


def call_gemini_for_project(model, interest: str, time_estimate: str) -> Optional[Dict]:
    """Ask Gemini for a beginner-level project idea. Return parsed dict or None on error."""
    prompt = (
        "Generate a beginner-level Python project idea for "
        f"{interest} theme, completable in {time_estimate}. "
        "Include: title, 100-word description, 3-5 steps, using only basic Python "
        "(loops, functions, lists, dicts, no external libraries). "
        "Keep it real-world, fun, and practical. "
        "Return JSON with keys: title, description, steps (array)."
    )
    try:
        response = model.generate_content(prompt)
        # Try to find JSON in response; some models may wrap it in text
        text = getattr(response, "text", None) or getattr(response, "candidates", None)
        if hasattr(response, "text"):
            content = response.text
        else:
            # Fallback attempt: stringify response; beginners can see raw text if needed
            content = str(response)
        # Try to parse JSON from the content
        data = None
        try:
            data = json.loads(content)
        except Exception:
            # Heuristic: extract JSON block if present
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start : end + 1]
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = None
        if not data:
            return None
        # Normalize fields
        title = str(data.get("title", "Untitled Project")).strip()
        description = str(data.get("description", "")).strip()
        steps_raw = data.get("steps", [])
        if isinstance(steps_raw, list):
            steps = [str(s).strip() for s in steps_raw if str(s).strip()]
        elif isinstance(steps_raw, str):
            steps = [s.strip() for s in steps_raw.split("\n") if s.strip()]
        else:
            steps = []
        if not title or not description or not steps:
            return None
        return {"title": title, "description": description, "steps": steps}
    except Exception as e:
        # Could be rate limits, invalid key, network errors, etc.
        print(f"Gemini error: {e}")
        return None


def call_gemini_for_feedback(model, project_title: str, code: str, syntax_ok: bool) -> Dict:
    """Ask Gemini for friendly feedback. Return dict with grade, tips, and echo of syntax result.

    Always returns a dict with safe defaults if model is None or errors occur.
    """
    safe_default = {
        "grade": "B",
        "tips": [
            "Use clear variable names.",
            "Add small comments above tricky parts.",
            "Test each function separately.",
        ],
        "syntax_ok": syntax_ok,
    }
    if not model:
        return safe_default
    prompt = (
        "Review this beginner Python code for "
        f"{project_title}: Check completeness (meets project steps), style (readable variable names, comments). "
        "Return JSON with keys: grade (A-F), tips (array of 3 short tips), syntax_ok (bool).\n\n"
        "Code:\n" + code
    )
    try:
        response = model.generate_content(prompt)
        content = getattr(response, "text", str(response))
        data = None
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(content[start : end + 1])
                except Exception:
                    data = None
        if not data:
            return safe_default
        grade = str(data.get("grade", "B")).strip()[:2]
        tips = data.get("tips", [])
        tips = [str(t).strip() for t in tips][:3] if isinstance(tips, list) else safe_default["tips"]
        syntax_flag = bool(data.get("syntax_ok", syntax_ok))
        return {"grade": grade, "tips": tips, "syntax_ok": syntax_flag}
    except Exception as e:
        print(f"Gemini feedback error: {e}")
        return safe_default


# -----------------------------
# Fallback beginner project ideas (hardcoded)
# -----------------------------
FALLBACK_IDEAS: List[Dict] = [
    {
        "title": "Number Guessing Game",
        "description": "Build a CLI game where the user guesses a random number (1-100), with hints.",
        "steps": [
            "Ask the user for a guess and compare to a secret number.",
            "Give higher or lower hints until correct.",
            "Count attempts and show a friendly message at the end.",
        ],
    },
    {
        "title": "To-Do List CLI",
        "description": "Create a text-based to-do list with add/remove tasks.",
        "steps": [
            "Display a menu to add, remove, and view tasks.",
            "Store tasks in a Python list.",
            "Save and load tasks to a simple text file (optional).",
        ],
    },
    {
        "title": "Budget Tracker",
        "description": "Track expenses in a list and calculate totals.",
        "steps": [
            "Let users add expenses with amount and category.",
            "Show total spent and totals per category.",
            "Export a summary as plain text (optional).",
        ],
    },
    {
        "title": "Text Adventure Game",
        "description": "Build a simple choose-your-own-adventure game.",
        "steps": [
            "Create rooms/scenes stored in dictionaries.",
            "Ask the player for choices to change scenes.",
            "Track inventory or score in a list (optional).",
        ],
    },
    {
        "title": "Grade Calculator",
        "description": "Compute average grades from user inputs.",
        "steps": [
            "Ask for a list of grades.",
            "Calculate average and letter grade.",
            "Handle invalid inputs politely.",
        ],
    },
]


# -----------------------------
# Utility functions
# -----------------------------

def save_project_to_db(project: Dict) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (title, description, steps, user_id) VALUES (?, ?, ?, ?)",
        (
            project["title"],
            project["description"],
            json.dumps(project["steps"]),
            SINGLE_USER_ID,
        ),
    )
    conn.commit()
    project_id = cur.lastrowid
    conn.close()
    return project_id


def get_project_from_db(project_id: int) -> Optional[Dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "steps": json.loads(row["steps"]),
        "user_id": row["user_id"],
    }


def get_user_profile() -> Dict:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(SUM(points), 0) FROM submissions WHERE user_id = ?",
        (SINGLE_USER_ID,),
    )
    total_points = cur.fetchone()[0]
    cur.execute(
        """
        SELECT p.id, p.title, MAX(s.timestamp) as last_time
        FROM projects p
        JOIN submissions s ON s.project_id = p.id AND s.user_id = ?
        GROUP BY p.id, p.title
        ORDER BY last_time DESC
        """,
        (SINGLE_USER_ID,),
    )
    completed = [{"project_id": r[0], "title": r[1], "last_time": r[2]} for r in cur.fetchall()]
    conn.close()
    return {"total_points": total_points, "completed": completed}


# -----------------------------
# Flask routes
# -----------------------------

@app.before_first_request
def setup_app() -> None:
    init_db()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Only one skill level for MVP; collect interest and time
        interest = request.form.get("interest", "").strip() or "General"
        time_estimate = request.form.get("time", "1-2 hours")

        model = configure_gemini_if_available()
        project = call_gemini_for_project(model, interest, time_estimate)

        if not project:
            # Pick a fallback idea and lightly adapt it to interest/time
            base = FALLBACK_IDEAS[0]
            # Simple rotation by current minute for variety
            idx = (datetime.utcnow().minute % len(FALLBACK_IDEAS))
            base = FALLBACK_IDEAS[idx]
            project = {
                "title": f"{base['title']} ({interest})",
                "description": base["description"],
                "steps": base["steps"],
            }

        project_id = save_project_to_db(project)
        return redirect(url_for("project_detail", project_id=project_id))

    return render_template("index.html")


@app.route("/project/<int:project_id>", methods=["GET", "POST"])
def project_detail(project_id: int):
    project = get_project_from_db(project_id)
    if not project:
        flash("Project not found.", "warning")
        return redirect(url_for("index"))

    qa_answer = None
    if request.method == "POST" and request.form.get("question") is not None:
        # Q&A mock response for now; future: integrate Gemini chat
        qa_answer = "Try breaking this into smaller functions."

    return render_template(
        "project.html",
        project=project,
        qa_answer=qa_answer,
    )


@app.route("/submit/<int:project_id>", methods=["POST"])
def submit_code(project_id: int):
    project = get_project_from_db(project_id)
    if not project:
        flash("Project not found.", "warning")
        return redirect(url_for("index"))

    code = request.form.get("code", "")
    # Syntax check using ast only (no execution)
    syntax_ok = False
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError as e:
        syntax_ok = False

    points = 10 if syntax_ok else 0

    # Get feedback from Gemini
    model = configure_gemini_if_available()
    feedback = call_gemini_for_feedback(model, project["title"], code, syntax_ok)

    # Store submission
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO submissions (user_id, project_id, code, timestamp, points) VALUES (?, ?, ?, ?, ?)",
        (
            SINGLE_USER_ID,
            project_id,
            code,
            datetime.utcnow().isoformat(),
            points,
        ),
    )
    conn.commit()
    conn.close()

    return render_template(
        "submission.html",
        project=project,
        syntax_ok=syntax_ok,
        points=points,
        feedback=feedback,
    )


@app.route("/profile")
def profile():
    data = get_user_profile()
    return render_template("profile.html", profile=data)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # For local dev: flask run will also work
    init_db()
    app.run(debug=True)
