"""
parser.py
---------
Everything to do with getting raw text off disk and into the agent.
No Gemini calls in here on purpose - this is pure file I/O so it can be
unit tested without an API key.
"""

import os

import config


class ResumeLoadError(Exception):
    pass


def load_resume(path: str) -> dict:
    """Load a single resume file and return a plain dict with its text.

    Supports .txt, .md, .pdf. Anything else raises ResumeLoadError so the
    caller can decide to skip it instead of crashing the whole batch.
    """
    if not os.path.exists(path):
        raise ResumeLoadError(f"File not found: {path}")

    filename = os.path.basename(path)
    candidate_id = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".txt", ".md"):
        text = _load_text_file(path)
    elif ext == ".pdf":
        text = _load_pdf_file(path)
    else:
        raise ResumeLoadError(f"Unsupported file type '{ext}' for {filename}")

    text = text.strip()
    if not text:
        raise ResumeLoadError(f"{filename} appears to be empty.")

    if len(text) > config.MAX_RESUME_CHARS:
        text = text[: config.MAX_RESUME_CHARS]

    return {
        "candidate_id": candidate_id,
        "filename": filename,
        "text": text,
    }


def load_resumes_from_dir(directory: str) -> tuple[list, list]:
    """Load every supported resume in a directory.

    Returns (loaded, errors) so the caller can report skipped files
    without the whole run failing because of one bad PDF.
    """
    loaded = []
    errors = []

    if not os.path.isdir(directory):
        raise ResumeLoadError(f"Resume directory not found: {directory}")

    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        full_path = os.path.join(directory, filename)
        if not os.path.isfile(full_path):
            continue
        try:
            resume = load_resume(full_path)
            loaded.append(resume)
        except ResumeLoadError as exc:
            errors.append({"filename": filename, "error": str(exc)})

        if len(loaded) >= config.MAX_CANDIDATES:
            break

    return loaded, errors


def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf_file(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ResumeLoadError(
            "pypdf is not installed. Run: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        text = "\n".join(pages_text).strip()
        if not text:
            raise ResumeLoadError(
                f"⚠ Could not extract text from candidate PDF: {os.path.basename(path)} "
                "(it may be a scanned image without a text layer)."
            )
        return text
    except ResumeLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ResumeLoadError(
            f"⚠ Could not extract text from candidate PDF: {os.path.basename(path)} ({exc})"
        ) from exc
