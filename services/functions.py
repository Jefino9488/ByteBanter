import pdfkit


def generate_pdf(html, filename):
    pdfkit.from_string(html, filename)
    return filename


