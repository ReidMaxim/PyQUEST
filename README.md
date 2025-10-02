## PyQuest (Flask) — Beginner Python Project Practice

PyQuest helps beginner Python learners generate simple, real-world project ideas and receive lightweight feedback using Google Gemini (free tier, `gemini-1.5-flash`). Build and test your code locally in VS Code or in Google Colab.

### Features
- Beginner-focused project idea generation (Gemini, with fallback ideas)
- Project details page with steps and a simple Q&A mock
- Code submission with syntax check (AST) and points
- Gemini feedback with grade and tips (fallback if API not available)
- Profile page with total points and completed projects

### Tech Stack
- Flask, SQLite
- Bootstrap 5 (via Bootstrap-Flask)
- Google Gemini API (`google-generativeai`)
- `.env` for configuration

### Setup
1. Python 3.8+
2. Create a virtual environment:
   - Linux/macOS:
     ```bash
     python -m venv env
     source env/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv env
     env\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env`:
   - Copy `.env.example` to `.env` and set:
     ```
     GEMINI_API_KEY=your_key_from_ai.google.dev
     ```
5. Run the app:
   ```bash
   flask run
   ```
   The app will be available at `http://localhost:5000`.

### Usage
- On the homepage, enter your interest (e.g., "Games", "Data Analysis") and choose a time estimate. Click Generate.
- Review the project details and steps. Ask a question (mocked response for now).
- Paste your Python code into the submission box. The app checks syntax only and assigns points (10 if valid, 0 if invalid).
- You will receive a grade and tips from Gemini (or a simple fallback response).
- Visit your profile to view total points and completed projects.

### Notes
- Paste project code into VS Code or Google Colab to run/test. Use Python 3.8+.
- Gemini free tier limits: ~15 RPM, 1M tokens/min, ~1,500 requests/day.
- This MVP uses a single user (no login). Future ideas: multi-user profiles, Q&A chat using Gemini, badges, leaderboards, richer project tracking.

### Project Structure
```
app.py
templates/
  index.html
  project.html
  submission.html
  profile.html
static/
  style.css
.env.example
requirements.txt
README.md
```

### Troubleshooting
- If the Gemini API key is missing or invalid, the app will fall back to built-in ideas and basic feedback. Check your `.env`.
- If `flask run` fails, ensure your virtual environment is activated and dependencies installed.

