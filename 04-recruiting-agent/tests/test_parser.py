import os, sys, shutil, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import parser

def test_load_txt_resume():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "candidate_01.txt")
        with open(path, "w") as f:
            f.write("Python, FastAPI, RAG experience.")
        resume = parser.load_resume(path)
        assert resume["candidate_id"] == "candidate_01"
    finally:
        shutil.rmtree(d)

def test_missing_file_raises():
    try:
        parser.load_resume("/tmp/does_not_exist_12345.txt")
        assert False
    except parser.ResumeLoadError:
        pass

def test_unsupported_extension_raises():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "c.docx")
        with open(path, "w") as f:
            f.write("x")
        try:
            parser.load_resume(path)
            assert False
        except parser.ResumeLoadError:
            pass
    finally:
        shutil.rmtree(d)

def _run_all():
    tests = [o for n, o in list(globals().items()) if n.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed")

if __name__ == "__main__":
    _run_all()
