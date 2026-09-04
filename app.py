import base64
import json
import os

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from database import save_project_analysis, get_project_from_db
from qr_generator import create_qr_code


app = Flask(__name__)
CORS(app)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SUPPORT = ["STRONG", "MODERATE", "WEAK", "UNSUPPORTED"]


# --------------------------------------------------
# GEMINI RESPONSE SCHEMA
# --------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": [
        "project",
        "technologies",
        "claims",
        "skills",
        "viva_questions",
        "portfolio"
    ],
    "properties": {
        "project": {
            "type": "OBJECT",
            "required": [
                "title",
                "problem",
                "solution",
                "project_story"
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
                "required": [
                    "name",
                    "evidence",
                    "strength"
                ],
                "properties": {
                    "name": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                    "strength": {
                        "type": "STRING",
                        "enum": SUPPORT
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
                    "reason"
                ],
                "properties": {
                    "claim": {"type": "STRING"},
                    "support": {
                        "type": "STRING",
                        "enum": SUPPORT
                    },
                    "supporting_evidence": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "missing_evidence": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "reason": {"type": "STRING"},
                },
            },
        },

        "skills": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "skill",
                    "strength",
                    "evidence"
                ],
                "properties": {
                    "skill": {"type": "STRING"},
                    "strength": {
                        "type": "STRING",
                        "enum": SUPPORT
                    },
                    "evidence": {"type": "STRING"},
                },
            },
        },

        "viva_questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "question",
                    "topic",
                    "reason"
                ],
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
                "evidence_highlights"
            ],
            "properties": {
                "summary": {"type": "STRING"},
                "contribution": {"type": "STRING"},
                "technologies": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "evidence_highlights": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
            },
        },
    },
}


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

REQUIRED_FIELDS = [
    "title",
    "problem",
    "solution",
    "contribution"
]


def validate(payload):

    if not isinstance(payload, dict):
        return False, "Body must be a JSON object."

    missing = [
        field
        for field in REQUIRED_FIELDS
        if not str(payload.get(field, "")).strip()
    ]

    if missing:
        return False, (
            "Missing required field(s): "
            + ", ".join(missing)
        )

    has_code = bool(
        str(payload.get("code", "")).strip()
    )

    has_images = bool(
        payload.get("screenshots")
    )

    if not has_code and not has_images:
        return False, (
            "Provide evidence: 'code' and/or "
            "'screenshots'."
        )

    return True, ""


# --------------------------------------------------
# GEMINI PROMPT
# --------------------------------------------------

def build_prompt(payload):

    claims = payload.get("claims") or []

    if claims:
        claims_block = (
            "Claims to verify:\n"
            + "\n".join(
                f"{i}. {claim}"
                for i, claim in enumerate(claims, 1)
            )
        )
    else:
        claims_block = (
            "No explicit claims were given. "
            "Extract the student's implicit claims "
            "from their stated contribution and solution, "
            "then verify each one against the evidence."
        )

    screenshot_note = (
        "Screenshot(s) are attached as images - "
        "use them as visual evidence.\n"
        if payload.get("screenshots")
        else ""
    )

    return f"""
You are an expert technical project evaluator for SkillProof.

Compare what the student CLAIMS they built against
the EVIDENCE they submitted.

Be strict and honest.

NON-NEGOTIABLE RULES:

- Base EVERY finding ONLY on the evidence provided.
- Never assume, extrapolate, or invent files,
  features, or facts.
- If evidence for a claim is absent, mark it
  WEAK or UNSUPPORTED.
- List a technology or skill only if a file,
  import, config, endpoint, or code line shows it.
- Every viva question MUST reference something
  concrete in THIS student's evidence.
- Do not create generic textbook questions.

SUPPORT LEVELS:

STRONG = evidence directly supports the claim.

MODERATE = supported but incomplete.

WEAK = some related evidence but insufficient.

UNSUPPORTED = no meaningful evidence.

STUDENT SUBMISSION

Project title:
{payload.get("title")}

Problem statement:
{payload.get("problem")}

Solution summary:
{payload.get("solution")}

Stated personal contribution:
{payload.get("contribution")}

{claims_block}

CODE / README / TECHNICAL EVIDENCE:

{payload.get("code", "(none provided)")}

{screenshot_note}

Analyze now and return the structured JSON.
"""


# --------------------------------------------------
# GEMINI CONTENT
# --------------------------------------------------

def build_contents(payload):

    from google.genai import types

    parts = [build_prompt(payload)]

    for shot in payload.get("screenshots", []) or []:

        if isinstance(shot, dict):
            data = shot.get("data", "")
            mime = shot.get(
                "mime_type",
                "image/png"
            )
        else:
            data = shot
            mime = "image/png"

        try:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(data),
                    mime_type=mime
                )
            )
        except Exception:
            continue

    return parts


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def get_client():

    from google import genai

    key = os.environ.get("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    return genai.Client(api_key=key)


# --------------------------------------------------
# HOME / FRONTEND
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return render_template("submit.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/healthz", methods=["GET"])
def health():

    return jsonify({
        "ok": True,
        "service": "skillproof",
        "model": MODEL
    })


# --------------------------------------------------
# GEMINI ANALYSIS
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    payload = request.get_json(
        silent=True
    )

    ok, msg = validate(payload)

    if not ok:
        return jsonify({
            "error": msg
        }), 400

    from google.genai import types

    try:

        client = get_client()

        response = client.models.generate_content(
            model=MODEL,
            contents=build_contents(payload),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
            ),
        )

    except RuntimeError as e:

        return jsonify({
            "error": str(e)
        }), 500

    except Exception as e:

        return jsonify({
            "error": f"Gemini request failed: {e}"
        }), 502

    try:

        raw_text = getattr(
            response,
            "text",
            ""
        ) or ""

        return jsonify(
            json.loads(raw_text)
        ), 200

    except Exception:

        return jsonify({
            "error": "Model did not return valid JSON.",
            "raw": str(
                getattr(
                    response,
                    "text",
                    ""
                )
            )
        }), 502


# --------------------------------------------------
# SAVE PROJECT + GENERATE QR
# --------------------------------------------------

@app.route(
    "/api/save-project",
    methods=["POST"]
)
def save_project():

    analysis = request.get_json(
        silent=True
    )

    if not analysis:

        return jsonify({
            "success": False,
            "error": "No analysis data provided"
        }), 400

    try:

        project_id = save_project_analysis(
            analysis
        )

        qr_path, public_url = create_qr_code(
            project_id
        )

        return jsonify({
            "success": True,
            "project_id": project_id,
            "public_url": public_url,
            "qr_url": f"/{qr_path}"
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# PUBLIC PORTFOLIO
# --------------------------------------------------

@app.route("/p/<project_id>")
def public_portfolio(project_id):

    project = get_project_from_db(
        project_id
    )

    if not project:

        return """
        <h1>Project Not Found</h1>
        <p>
        The requested SkillProof project
        does not exist.
        </p>
        """, 404

    analysis = project.get(
        "analysis",
        {}
    )

    return render_template(
        "public_portfolio.html",
        project=project,
        analysis=analysis,
        project_id=project_id
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )