from PIL import Image
import pytesseract
from memory.qdrant_store import store_fact


def store_image_memory(image_path: str, memory_type: str = "scan"):
    """
    Extracts text from an image and stores it as medical memory
    """

    # 1️⃣ Load image
    image = Image.open(image_path)

    # 2️⃣ Extract text using OCR
    extracted_text = pytesseract.image_to_string(image)

    if not extracted_text.strip():
        extracted_text = "Medical image uploaded, but no readable text detected."

    # 3️⃣ Store extracted info as memory
    store_fact(extracted_text, memory_type)

    return extracted_text
