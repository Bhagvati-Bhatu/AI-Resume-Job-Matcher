import pymupdf


def extract_text_from_pdf(file_path):

    document = pymupdf.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def extract_text_from_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text