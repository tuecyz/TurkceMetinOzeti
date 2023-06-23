import openai
import re
from collections import Counter
import pdfplumber

openai.api_key = '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'

def summarize_paper(paper_file_path):
    paper_content = pdfplumber.open(paper_file_path).pages

    summarized_text = ""

    for page in paper_content:
        text = page.extract_text()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=100,  # Adjust the maximum number of tokens for the summarized text
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        if 'choices' in response and len(response.choices) > 0:
            summary = response.choices[0].text.strip()
            summarized_text += summary + " "
        else:
            summarized_text += "Özetleme yapılamadı."

    return summarized_text
