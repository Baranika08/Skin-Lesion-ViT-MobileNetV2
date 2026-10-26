"""ONE command for everything:  python run_all.py
 1) self-test  2) experiments  3) figures  4) report (DOCX + PDF) and slides (PPTX) in ./documents
Options:  --dataset ham10000 --data-root data/HAM10000   (real data)   |   --preset quick|cpu|standard"""
import subprocess, sys

args = sys.argv[1:]
print("== Step 1/3: self-test ==")
if subprocess.run([sys.executable, "-u", "selftest.py"]).returncode != 0:
    sys.exit("Self-test failed. Check that you ran:  pip install -r requirements.txt  (copy the error above to get help).")
print("\n== Step 2/3: experiments (this is the long step) ==")
subprocess.run([sys.executable, "-u", "run_experiments.py", *args], check=True)
print("\n== Step 3/3: building report, PDF and slides ==")
subprocess.run([sys.executable, "-u", "make_documents.py"], check=True)
print("\nFINISHED. Open the 'documents' folder (report .docx/.pdf and slides .pptx).")
