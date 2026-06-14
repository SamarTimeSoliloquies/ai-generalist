import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import load_pdf

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "uploaded_docs")
domains = ["law", "accounting", "medicine"]

for domain in domains:
    folder = os.path.join(DOCS_FOLDER, domain)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not files:
        print(f"No PDFs found in {domain} folder.")
        continue

    print(f"\nUploading {len(files)} files for domain: {domain}")
    for filename in files:
        path = os.path.join(folder, filename)
        print(f"  Processing: {filename}...")
        chunks = load_pdf(path, domain)
        if chunks == 0:
            print(f"  Already in database, skipped.")
        else:
            print(f"  Done — {chunks} chunks created.")

print("\nAll done!")