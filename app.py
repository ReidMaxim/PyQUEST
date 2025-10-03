import os
import json
import sqlite3
import ast
import random
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap5
from dotenv import load_dotenv

# Optional import: google-generativeai (Gemini). Keep the app working if the package/key is missing.
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None


# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
Bootstrap5(app)

# For a simple MVP, we'll assume a single user (user_id = 1) with no auth.
SINGLE_USER_ID = 1

# SQLite database configuration. This creates a database file in the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "pyquest.db")


def get_db_connection() -> sqlite3.Connection:
    """Create or return a SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they don't exist yet.

    Tables:
    - projects(id, title, description, steps(json), user_id)
    - submissions(id, user_id, project_id, code, timestamp, points)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            steps TEXT NOT NULL,
            user_id INTEGER NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            points INTEGER NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
        """
    )

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Gemini configuration
# ----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Create a model instance if the package and key are available; otherwise keep None.
GEMINI_MODEL = None
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        GEMINI_MODEL = None


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_fallback_ideas() -> list[dict]:
    """Return a small catalog of beginner-friendly fallback project ideas.

    Each idea is a dict with keys: title, description, steps(list[str]).
    """
    return [
        {
            "title": "Number Guessing Game",
            "description": (
                "Build a CLI game where the user guesses a random number between 1 and 100. "
                "Provide hints like 'too high' or 'too low' and track attempts."
            ),
            "steps": [
                "Generate a random number between 1 and 100.",
                "Ask the user to guess until they get it right.",
                "Print hints and count the number of attempts.",
                "Offer to play again at the end.",
            ],
        },
        {
            "title": "To-Do List CLI",
            "description": (
                "Create a text-based to-do list that supports adding, listing, and removing tasks. "
                "Store tasks in a list while the program runs."
            ),
            "steps": [
                "Show a simple menu: add, list, remove, quit.",
                "Use a list to store tasks in memory.",
                "Implement functions for each menu option.",
                "Validate user input and print helpful messages.",
            ],
        },
        {
            "title": "Budget Tracker",
            "description": (
                "Track expenses and incomes in a list of dictionaries and calculate totals. "
                "Print a simple summary of spending by category."
            ),
            "steps": [
                "Allow adding entries with amount and category.",
                "Store entries in a list of dictionaries.",
                "Compute totals and balances using loops.",
                "Print a simple report at the end.",
            ],
        },
        {
            "title": "Text Adventure Game",
            "description": (
                "Build a simple choose-your-own-adventure game in the terminal with a few scenes "
                "and choices that lead to different outcomes."
            ),
            "steps": [
                "Create functions for each scene.",
                "Use input() to get choices from the player.",
                "Use if/elif to branch the story.",
                "End with a win/lose message and replay option.",
            ],
        },
        {
            "title": "Grade Calculator",
            "description": (
                "Ask the user for multiple grades, store them in a list, and compute the average. "
                "Print the final grade and a simple message."
            ),
            "steps": [
                "Collect grades using input() in a loop.",
                "Store numbers in a list and validate input.",
                "Compute average with sum() and len().",
                "Print the result formatted to two decimals.",
            ],
        },
    ]


def parse_gemini_project(text: str) -> dict | None:
    """Attempt to parse the Gemini response as a project dict.

    We ask Gemini to return JSON, but if not strictly JSON, try a forgiving strategy.
    Returns a dict with keys: title, description, steps (list[str]) or None if parsing fails.
    """
    if not text:
        return None

    # First try: strict JSON detection between the first '{' and last '}'.
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            title = str(obj.get("title", "")).strip()
            description = str(obj.get("description", "")).strip()
            steps_raw = obj.get("steps", [])
            steps: list[str] = [str(s).strip() for s in steps_raw if str(s).strip()]
            if title and description and steps:
                return {"title": title, "description": description, "steps": steps}
    except Exception:
        pass

    # Second try: heuristic parsing if the response is markdown-like.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = ""
    description = ""
    steps: list[str] = []
    mode: str | None = None
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("title:"):
            mode = None
            title = ln.split(":", 1)[1].strip()
        elif lower.startswith("description:"):
            mode = "description"
            description = ln.split(":", 1)[1].strip()
        elif lower.startswith("steps:"):
            mode = "steps"
        elif mode == "description":
            description += (" " if description else "") + ln
        elif mode == "steps":
            # Try to strip list markers like '-', '*', '1.'
            cleaned = ln.lstrip("-*0123456789. ")
            if cleaned:
                steps.append(cleaned)

    if title and description and steps:
        return {"title": title, "description": description, "steps": steps}

    return None


def generate_project_idea(interest: str, time_estimate: str) -> dict:
    """Generate a beginner-level Python project idea using Gemini, with fallback.

    The returned dict has keys: title, description, steps(list[str]).
    """
    prompt = (
        "Generate a beginner-level Python project idea for "
        f"{interest} theme, completable in {time_estimate}. "
        "Include: title, 100-word description, 3-5 steps, using only basic Python "
        "(loops, functions, lists, dicts, no external libraries). Keep it real-world, fun, and practical.\n\n"
        "Return JSON with keys exactly: {\"title\": str, \"description\": str, \"steps\": list}."
    )

    # Try Gemini first if available.
    if GEMINI_MODEL is not None:
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            text = getattr(response, "text", None)
            project = parse_gemini_project(text or "")
            if project:
                return project
        except Exception as e:  # Handle invalid keys, rate limits, etc.
            # For beginners, we keep failures gentle and move to a fallback idea.
            print(f"[Gemini Error] {e}")

    # Fallback to one of the hardcoded ideas and lightly tailor text to the interest/time.
    idea = random.choice(get_fallback_ideas())
    tailored = {
        "title": f"{idea['title']} ({interest})",
        "description": (
            f"{idea['description']} This is designed for the '{interest}' theme "
            f"and should be completable in {time_estimate}."
        ),
        "steps": idea["steps"],
    }
    return tailored


def grade_with_gemini(project_title: str, code: str, syntax_ok: bool) -> dict:
    """Use Gemini to provide feedback. If unavailable, return a simple heuristic result.

    Expected return keys: grade(str), tips(list[str]), syntax_ok(bool), feedback(str)
    """
    if GEMINI_MODEL is None:
        # Simple offline fallback for beginners.
        grade = "B" if syntax_ok else "D"
        tips = [
            "Use readable variable names.",
            "Add comments to explain key steps.",
            "Break long code into small functions.",
        ]
        feedback = (
            "Nice start! Focus on clarity and small functions. If something fails, "
            "test smaller parts first."
        )
        return {"grade": grade, "tips": tips, "syntax_ok": syntax_ok, "feedback": feedback}

    prompt = (
        "Review this beginner Python code for the project titled '"
        f"{project_title}" "':\n\n"
        f"{code}\n\n"
        "Check completeness (meets project steps) and style (readable names, comments). "
        "Return JSON with keys exactly: {\"grade\": one of [A,B,C,D,F], \"tips\": list of 3 short tips, \"syntax_ok\": true/false, \"feedback\": short paragraph}. "
        f"Confirm syntax result = {str(syntax_ok).lower()} in your JSON."
    )

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = getattr(response, "text", None) or ""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            grade = str(data.get("grade", "B")).strip() or "B"
            tips_raw = data.get("tips", [])
            tips = [str(t).strip() for t in tips_raw if str(t).strip()]
            fb = str(data.get("feedback", "")).strip()
            syn = data.get("syntax_ok", syntax_ok)
            if not tips:
                tips = [
                    "Use readable variable names.",
                    "Add comments to explain key steps.",
                    "Break long code into small functions.",
                ]
            if not fb:
                fb = "Thanks for submitting your code! Keep iterating and testing."
            return {"grade": grade, "tips": tips, "syntax_ok": bool(syn), "feedback": fb}
    except Exception as e:
        print(f"[Gemini Feedback Error] {e}")

    # Fallback if parsing or API fails.
    grade = "B" if syntax_ok else "D"
    tips = [
        "Use readable variable names.",
        "Add comments to explain key steps.",
        "Break long code into small functions.",
    ]
    feedback = (
        "Great effort! Improve structure by using functions and adding comments. "
        "Test each part step-by-step."
    )
    return {"grade": grade, "tips": tips, "syntax_ok": syntax_ok, "feedback": feedback}


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.before_first_request
def _startup() -> None:
    """Initialize database on first request to keep setup simple for beginners."""
    init_db()


@app.route("/", methods=["GET"])
def index():
    """Homepage with a simple form to request a project idea."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a project idea using Gemini (with fallback) and store it in SQLite."""
    skill = request.form.get("skill", "Beginner").strip()
    interest = request.form.get("interest", "").strip() or "General"
    time_estimate = request.form.get("time", "1-2 hours").strip()

    # For MVP, we only allow Beginner, but keep the field for future expansion.
    if skill.lower() != "beginner":
        skill = "Beginner"

    project = generate_project_idea(interest, time_estimate)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects(title, description, steps, user_id) VALUES (?, ?, ?, ?)",
        (
            project["title"],
            project["description"],
            json.dumps(project["steps"]),
            SINGLE_USER_ID,
        ),
    )
    project_id = cur.lastrowid
    conn.commit()
    conn.close()

    return redirect(url_for("project", project_id=project_id))


@app.route("/project/<int:project_id>", methods=["GET"])
def project(project_id: int):
    """Show project details, Q&A input (mocked), and code submission form."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return redirect(url_for("index"))

    steps = []
    try:
        steps = json.loads(row["steps"]) or []
    except Exception:
        pass

    return render_template(
        "project.html",
        project={
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "steps": steps,
        },
        qa_response=None,
    )


@app.route("/ask/<int:project_id>", methods=["POST"])
def ask(project_id: int):
    """Mock a Q&A response for a beginner. Placeholder for future Gemini chat integration."""
    _question = request.form.get("question", "").strip()

    # Future expansion: integrate Gemini API here for a real Q&A chat experience.
    mocked_response = "Try breaking this into smaller functions."

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return redirect(url_for("index"))

    steps = []
    try:
        steps = json.loads(row["steps"]) or []
    except Exception:
        pass

    return render_template(
        "project.html",
        project={
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "steps": steps,
        },
        qa_response=mocked_response,
    )


@app.route("/submit/<int:project_id>", methods=["POST"])
def submit(project_id: int):
    """Receive code, run a basic AST syntax check, assign points, and get Gemini feedback."""
    code = request.form.get("code", "")

    # Syntax check: parse the code into an AST. If it fails, it's invalid syntax.
    syntax_ok = True
    try:
        ast.parse(code)
    except Exception:
        syntax_ok = False

    points = 10 if syntax_ok else 0

    # Get the project to enrich the feedback.
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT title FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    project_title = row["title"] if row else "Your Project"

    # Store the submission in the database.
    timestamp = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO submissions(user_id, project_id, code, timestamp, points) VALUES (?, ?, ?, ?, ?)",
        (SINGLE_USER_ID, project_id, code, timestamp, points),
    )
    conn.commit()
    conn.close()

    # Ask Gemini for feedback (with robust fallbacks).
    feedback = grade_with_gemini(project_title, code, syntax_ok)

    return render_template(
        "submission.html",
        project_title=project_title,
        code=code,
        points=points,
        grade=feedback.get("grade", "B"),
        tips=feedback.get("tips", []),
        syntax_ok=feedback.get("syntax_ok", syntax_ok),
        feedback=feedback.get("feedback", "Thanks for your submission!"),
    )


@app.route("/profile", methods=["GET"])
def profile():
    """Show total points and list of completed projects."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COALESCE(SUM(points), 0) AS total_points FROM submissions WHERE user_id = ?",
        (SINGLE_USER_ID,),
    )
    row = cur.fetchone()
    total_points = int(row["total_points"]) if row else 0

    cur.execute(
        """
        SELECT DISTINCT p.id, p.title
        FROM projects p
        JOIN submissions s ON s.project_id = p.id
        WHERE s.user_id = ? AND s.points > 0
        ORDER BY s.timestamp DESC
        """,
        (SINGLE_USER_ID,),
    )
    completed = cur.fetchall()

    conn.close()

    projects = [{"id": r["id"], "title": r["title"]} for r in completed]
    return render_template("profile.html", total_points=total_points, completed_projects=projects)


if __name__ == "__main__":
    # Running via `flask run` is recommended. This is for direct execution support.
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)

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
