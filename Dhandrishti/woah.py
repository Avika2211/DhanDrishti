import cv2
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image_path = "Pictures\Screenshots\Screenshot 2024-04-11 204716.png"  # Yahan apni image ka path do
image = cv2.imread(image_path)

# Convert to grayscale (optional, but helps in OCR accuracy)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR
extracted_text = pytesseract.image_to_string(gray)

print("Extracted Text:")
print(extracted_text)
