import requests
from PyPDF2 import PdfReader

def download_pdf(url, save_path="temp.pdf"):
    r = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(r.content)
    return save_path

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
