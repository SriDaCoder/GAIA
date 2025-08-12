from PyPDF2 import PdfReader, PdfWriter

# Load PDFs
main_pdf = PdfReader(r"C:\Users\srini\OneDrive - GEMS EDUCATION\Documents\BST-HHW.pdf")
extra_pdf = PdfReader(r"C:\Users\srini\Downloads\BST-HHW-main.pdf")

# Create a writer
writer = PdfWriter()

# Add all pages from the main PDF
for page in main_pdf.pages:
    writer.add_page(page)

# Add a specific page from the extra PDF (e.g., first page = index 0)
writer.add_page(extra_pdf.pages[-1])

# Save the result
with open("merged.pdf", "wb") as f:
    writer.write(f)
