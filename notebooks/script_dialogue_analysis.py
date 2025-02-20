import os
import PyPDF2
import re
from bs4 import BeautifulSoup

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_number in range(reader.numPages):
            text += reader.getPage(page_number).extractText()
    return text

# Function to convert text to HTML
def text_to_html(text):
    #I've assumed the script to be preprocessed and fed for simplicity of the model
    # Converting text to HTML
    html = '{}'.format(text)
    return html

# Function to swap gender-specific pronouns and names
def swap_gender_context(dialogue):
    # I'm replacing gender-specific pronouns and names
    dialogue = re.sub(r'\b(he|him|his)\b', 'she', dialogue, flags=re.IGNORECASE)
    dialogue = re.sub(r'\b(himself)\b', 'herself', dialogue, flags=re.IGNORECASE)
    dialogue = re.sub(r'\b(his)\b', 'her', dialogue, flags=re.IGNORECASE)
    dialogue = re.sub(r'\b(boy|man|actor)\b', 'girl', dialogue, flags=re.IGNORECASE)
    dialogue = re.sub(r'\b(male)\b', 'female', dialogue, flags=re.IGNORECASE)
    dialogue = re.sub(r'\b(men|guys|boys|actors)\b', 'girls', dialogue, flags=re.IGNORECASE)
    # We can add mode pronouns specifing gender as required
    return dialogue

# Function to extract dialogues for a given cast member
def extract_dialogues(html, cast_member):
    soup = BeautifulSoup(html, 'html.parser')
    dialogues = []
    for element in soup.find_all('p'):
        # I'm assuming dialogues are within  tags
        dialogue_text = element.get_text()
        if cast_member in dialogue_text:
            dialogues.append(swap_gender_context(dialogue_text))
    return dialogues

# Performing analysis on the extracted dialogues
def perform_analysis(dialogues):
    pass

def main(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            html = text_to_html(text)

            cast_member = "John"
            dialogues = extract_dialogues(html, cast_member)

            # Performing analysis on the extracted dialogues by calling function
            perform_analysis(dialogues)

input_dir = "/content/drive/MyDrive/Bollywood-Data-master/scripts-data"
main(input_dir)
