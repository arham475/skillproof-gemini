import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from zipfile import ZipFile

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from database import save_project_analysis, get_project_from_db
from qr_generator import create_qr_code


app = Flask(__name__)
CORS(app)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SUPPORT = ["STRONG", "MODERATE", "WEAK", "UNSUPPORTED"]


# =========================================================
# GEMINI RESPONSE SCHEMA
# =========================================================

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": [
        "project",
        "technologies",
        "claims",
        "skills",
        "viva_questions",
        "portfolio",
    ],
    "properties": {
        "project": {
            "type": "OBJECT",
            "required": [
                "title",
                "problem",
                "solution",
                "project_story",
            ],
            "properties": {
                "title": {"type": "STRING"},
                "problem": {"type": "STRING"},
                "solution": {"type": "STRING"},
                "project_story": {"type": "STRING"},
            },
        },
        "technologies": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["name", "evidence", "strength"],
                "properties": {
                    "name": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                    "strength": {
                        "type": "STRING",
                        "enum": SUPPORT,
                    },
                },
            },
        },
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "claim",
                    "support",
                    "supporting_evidence",
                    "missing_evidence",
                    "reason",
                ],
                "properties": {
                    "claim": {"type": "STRING"},
                    "support": {
                        "type": "STRING",
                        "enum": SUPPORT,
                    },
                    "supporting_evidence": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "missing_evidence": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "reason": {"type": "STRING"},
                },
            },
        },
        "skills": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["skill", "strength", "evidence"],
                "properties": {
                    "skill": {"type": "STRING"},
                    "strength": {
                        "type": "STRING",
                        "enum": SUPPORT,
                    },
                    "evidence": {"type": "STRING"},
                },
            },
        },
        "viva_questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["question", "topic", "reason"],
                "properties": {
                    "question": {"type": "STRING"},
                    "topic": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
            },
        },
        "portfolio": {
            "type": "OBJECT",
            "required": [
                "summary",
                "contribution",
                "technologies",
                "evidence_highlights",
            ],
            "properties": {
                "summary": {"type": "STRING"},
                "contribution": {"type": "STRING"},
                "technologies": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "evidence_highlights": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
        },
    },
}


# =========================================================
# VALIDATION
# =========================================================

def validate(payload):
    if not isinstance(payload, dict):
        return False, "Request body must be JSON."

    required = [
        "problem",
        "solution",
        "contribution",
    ]

    missing = [
        field
        for field in required
        if not str(payload.get(field, "")).strip()
    ]

    if missing:
        return False, "Missing: " + ", ".join(missing)

    repo_url = str(payload.get("repo_url", "")).strip()
    code = str(payload.get("code", "")).strip()

    if not repo_url and not code:
        return False, "Provide a GitHub repository URL."

    return True, ""


# =========================================================
# GITHUB
# =========================================================

def parse_github_url(repo_url):
    repo_url = repo_url.strip().rstrip("/")

    match = re.match(
        r"^https?://github\.com/([^/]+)/([^/#]+)",
        repo_url,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Please enter a valid public GitHub repository URL."
        )

    owner = match.group(1)
    repo = match.group(2)

    if repo.endswith(".git"):
        repo = repo[:-4]

    return owner, repo


def github_request(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SkillProof/1.0",
            "Accept": "application/vnd.github+json",
        },
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()


def fetch_github_evidence(repo_url):
    owner, repo = parse_github_url(repo_url)

    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}"
    )

    try:
        repo_info = json.loads(
            github_request(api_url).decode("utf-8")
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                "GitHub repository was not found or is private."
            )
        raise ValueError(
            f"GitHub returned HTTP {e.code}."
        )

    branch = repo_info.get("default_branch", "main")

    zip_url = (
        f"https://github.com/{owner}/{repo}"
        f"/archive/refs/heads/{branch}.zip"
    )

    try:
        zip_bytes = github_request(zip_url)
    except Exception as e:
        raise ValueError(
            f"Could not download GitHub repository: {e}"
        )

    # Protect the server from huge repositories.
    if len(zip_bytes) > 15 * 1024 * 1024:
        raise ValueError(
            "Repository is too large. Use a smaller public repository."
        )

    allowed_extensions = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".html",
        ".css",
        ".sql",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".md",
        ".txt",
        ".env.example",
    }

    ignored_parts = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
    }

    output = []
    total_chars = 0
    file_count = 0

    try:
        with ZipFile(io.BytesIO(zip_bytes)) as archive:
            names = archive.namelist()

            for name in names:
                if file_count >= 80:
                    break

                clean_name = name.replace("\\", "/")
                parts = clean_name.split("/")

                if any(
                    part in ignored_parts
                    for part in parts
                ):
                    continue

                filename = parts[-1]

                if not filename:
                    continue

                lower_name = filename.lower()

                if not (
                    any(
                        lower_name.endswith(ext)
                        for ext in allowed_extensions
                    )
                    or lower_name in {
                        "dockerfile",
                        "makefile",
                    }
                ):
                    continue

                try:
                    raw = archive.read(name)

                    if b"\x00" in raw[:4096]:
                        continue

                    text = raw.decode(
                        "utf-8",
                        errors="ignore",
                    )

                except Exception:
                    continue

                remaining = 120000 - total_chars

                if remaining <= 0:
                    break

                text = text[:remaining]

                output.append(
                    f"\n===== {clean_name} =====\n{text}\n"
                )

                total_chars += len(text)
                file_count += 1

    except Exception as e:
        raise ValueError(
            f"Could not read repository archive: {e}"
        )

    if not output:
        raise ValueError(
            "No readable source files were found."
        )

    header = (
        f"GitHub repository: {owner}/{repo}\n"
        f"Default branch: {branch}\n"
        f"Files inspected: {file_count}\n"
    )

    return header + "".join(output)


# =========================================================
# GEMINI PROMPT
# =========================================================

def build_prompt(payload, github_evidence=""):
    claims = payload.get("claims") or []

    if claims:
        claims_block = "\n".join(
            f"{i}. {claim}"
            for i, claim in enumerate(claims, 1)
        )
    else:
        claims_block = (
            "Extract the important technical claims from "
            "the student's solution and contribution."
        )

    return f"""
You are SkillProof, an evidence-based technical project
verification system.

Your job is NOT to blindly trust the student's claims.

Compare every claim against the actual submitted
repository evidence.

STRICT RULES:

1. Use ONLY evidence contained in the repository.
2. Never invent files, technologies, APIs, tests,
   algorithms, or features.
3. If a claim is not supported by repository evidence,
   classify it WEAK or UNSUPPORTED.
4. A technology is supported only when code, imports,
   dependencies, configuration, or another concrete
   repository artifact proves it.
5. Viva questions must reference the student's actual
   repository implementation.
6. Do not generate generic textbook viva questions.
7. Clearly identify unsupported claims.

SUPPORT LEVELS:

STRONG:
Direct implementation evidence exists.

MODERATE:
Relevant evidence exists but implementation is incomplete.

WEAK:
Some related evidence exists but it is insufficient.

UNSUPPORTED:
No meaningful repository evidence supports the claim.

PROJECT TITLE:
{payload.get("title", "")}

PROBLEM:
{payload.get("problem", "")}

SOLUTION:
{payload.get("solution", "")}

STUDENT CONTRIBUTION:
{payload.get("contribution", "")}

CLAIMS:
{claims_block}

ACTUAL GITHUB REPOSITORY EVIDENCE:
{github_evidence}

ADDITIONAL SUBMITTED CODE:
{payload.get("code", "")}

Return the complete structured JSON.
"""


# =========================================================
# GEMINI
# =========================================================

def get_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server."
        )

    return genai.Client(api_key=key)


def analyze_with_gemini(payload, github_evidence):
    from google.genai import types

    client = get_client()

    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(
            payload,
            github_evidence,
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    raw = getattr(response, "text", "") or ""

    if not raw:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return json.loads(raw)


# =========================================================
# FRONTEND
# =========================================================

@app.route("/")
def home():
    return render_template("submit.html")


@app.route("/analysis.html")
def analysis_page():
    project_id = request.args.get("id", "")
    return render_template(
        "analysis.html",
        project_id=project_id,
    )


@app.route("/portfolio.html")
def portfolio_page():
    project_id = request.args.get("id", "")
    return render_template(
        "portfolio.html",
        project_id=project_id,
    )


# =========================================================
# HEALTH
# =========================================================

@app.route("/healthz")
def health():
    return jsonify({
        "ok": True,
        "service": "skillproof",
        "model": MODEL,
    })


# =========================================================
# ANALYZE
# =========================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True)

    ok, message = validate(payload)

    if not ok:
        return jsonify({
            "success": False,
            "error": message,
        }), 400

    try:
        repo_url = str(
            payload.get("repo_url", "")
        ).strip()

        github_evidence = ""

        if repo_url:
            github_evidence = fetch_github_evidence(
                repo_url
            )

        analysis = analyze_with_gemini(
            payload,
            github_evidence,
        )

        # Save the repository URL inside the analysis
        # so the portfolio can display it.
        analysis["_skillproof"] = {
            "repo_url": repo_url,
            "model": MODEL,
        }

        project_id = save_project_analysis(
            analysis
        )

        qr_path, public_url = create_qr_code(
            project_id
        )

        return jsonify({
            "success": True,
            "project_id": project_id,
            "analysis": analysis,
            "public_url": public_url,
            "qr_url": f"/{qr_path}",
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500

    except Exception as e:
        print("ANALYSIS ERROR:", repr(e))

        return jsonify({
            "success": False,
            "error": f"Analysis failed: {e}",
        }), 502


# =========================================================
# PROJECT API
# =========================================================

@app.route(
    "/api/project/<project_id>",
    methods=["GET"],
)
def project_api(project_id):
    try:
        project = get_project_from_db(
            project_id
        )

        if not project:
            return jsonify({
                "success": False,
                "error": "Project not found.",
            }), 404

        return jsonify({
            "success": True,
            "project": project,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =========================================================
# PUBLIC PORTFOLIO
# =========================================================

@app.route("/p/<project_id>")
def public_portfolio(project_id):
    project = get_project_from_db(
        project_id
    )

    if not project:
        return """
        <h1>Project Not Found</h1>
        <p>This SkillProof project does not exist.</p>
        """, 404

    analysis = project.get(
        "analysis",
        {},
    )

    return render_template(
        "public_portfolio.html",
        project=project,
        analysis=analysis,
        project_id=project_id,
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),
    )