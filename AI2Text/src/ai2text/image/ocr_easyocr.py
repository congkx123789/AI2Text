import easyocr

def extract_text_easyocr(image_path, lang="en"):
    reader = easyocr.Reader([lang])
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)
