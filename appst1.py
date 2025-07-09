

import fitz
import streamlit as st

from pdf2image import convert_from_bytes

from PIL import Image
import pytesseract
import re
import io
import pandas as pd
from transformers import AutoTokenizer
from fpdf import FPDF
import google.generativeai as genai

st.set_page_config(page_title="PDF Comparator", layout="wide")
st.title("Synthite AI - QA Specification & Regulation Checker")
# Configure Gemini API

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

def count_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def ocr_extract_text(pdf_file, psm=6):
    images = convert_from_bytes(pdf_file.read(), dpi=300)
    all_text = []
    for img in images:
        custom_config = f"--psm {psm}"
        text = pytesseract.image_to_string(img, config=custom_config)
        cleaned = postprocess_ocr(text)
        all_text.append(cleaned)
    return "\n\n".join(all_text)

def extract_text(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

def postprocess_ocr(text):
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 1]
    merged_lines = []
    skip = False
    for i in range(len(lines)):
        if skip:
            skip = False
            continue
        current = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.search(r"[a-zA-Z0-9)]$", current) and re.match(r"^[<>=~±0-9µ%a-zA-Z /.-]+$", next_line):
                merged_lines.append(f"{current}: {next_line}")
                skip = True
                continue
        merged_lines.append(current)
    return "\n".join(merged_lines)

def emphasize_params(text):
    return re.sub(r"(?i)([A-Za-z /()°µ%-]+)[\-:–—]+ ?([^\n]+)", r"🔹 \1: \2", text)

def compare_documents(req_text, pis_text):
    req_tokens = count_tokens(req_text)
    pis_tokens = count_tokens(pis_text)
    print(f"\n🔢 Token size of Requirement File: {req_tokens}")
    print(f"🔢 Token size of Product Info File: {pis_tokens}")
    prompt = f"""
Compare the following two technical documents as a Quality analyst:

--- Requirement Sheet ---
{req_text}

--- PIS Sheet ---
{pis_text}

Instructions:
1. List every field from requirement sheet (req_text) in the output table without fail no matter how much the table length.
2. List all parameters that match exactly (value and meaning).
3. List parameters that are semantically similar (e.g., "Aroma" vs "Odour") with values within acceptable limits.
4. List all mismatches (missing, out of range, or incorrect).
5. Identify if both refer to the same or similar meaning, even with different wording.
6. Show extra parameters found in one but not the other.
7. Consider plural words and partially matched words as same.
8.For each regulation you mention (e.g., EU, ASTA, ISO, EC, BAM, AOAC,JECFA,etc.), include a hyperlink to the official regulation page or source if known.
8.1. for sites like ISO dont pick the first site comes unless until its Online Browsing Platform (OBP) from iso.org ,for ISO 4883-2 : ISO regulations are listed like "https://www.iso.org/obp/ui/en/#iso:std:iso:4833:-2:ed-1:v1:en" so do like this
8.2. Incase of FCC regulations not the entire FCC site but  try to pick the exact edition as mentioned  for FCC 12th Ed. : https://www.foodchemicalscodex.org/sites/fcconline/files/usp_pdf/EN/fcc/fcc-12-commentary-20200302.pdf take there directly
8.3. for JECFA : https://www.fao.org/food/food-safety-quality/scientific-advice/jecfa/jecfa-additives/detail/en/c/358/
8.4 incase if the extracted regulation is  AOAC regulations not the entire AOAC website but try to pick the exact edition as mentioned for AOAC 19th Ed. 999.11 : https://academic.oup.com/aoac-publications/search-results?f_BookID=45491&fl_SiteID=6535&fl_BookID=45491&cqb=[22terms%22:[%22filter%22:%22%22,%22input%22:%22AOAC%20999.11%22]]&qb=%22q%22:%22AOAC%20999.11%22&page=1
8.5 incase if the extracted regulation is ASTA if Regulations is ASTA 24.2, 1997 or ASTA 24.2 then fetch  https://astaspice.org/resources/search?q=ASTA%2024.2 like this

9.If the exact link is unavailable, suggest the most likely official website or portal where the regulation can be found ( europa.eu for EU regulations, etc.).



Output in table format with columns:
| Parameter | Requirement Value | PIS Value | Regulations or Method in Requirement | Regulations or Method in PIS | Match Status (✅ / ⚠️ / ❌) | Reason |
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.4))
    result_text = response.text.strip()
    response_tokens = count_tokens(result_text)
    total_tokens = req_tokens + pis_tokens + response_tokens

    print(f"🧮 Total Tokens Used: {total_tokens}")
    return result_text, total_tokens

def parse_markdown_table(markdown_text):
    lines = markdown_text.strip().split('\n')
    table_lines = [line for line in lines if line.strip().startswith('|')]

    if len(table_lines) < 3:
        st.warning("⚠️ No valid table found in the markdown.")
        return pd.DataFrame()

    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip('|').split('|')]
    expected_cols = len(headers)

    valid_rows = []
    for line in table_lines[2:]:
        row = [cell.strip() for cell in line.strip('|').split('|')]
        if len(row) == expected_cols:
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows, columns=headers)
    return df

# def create_pdf_from_text(text_response):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Helvetica", size=12)

#     lines = text_response.split('\n')
#     for line in lines:
#         pdf.cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)

#     pdf_bytes = pdf.output(dest='S').encode('latin-1')
#     return pdf_bytes

from fpdf import FPDF, XPos, YPos

def create_pdf_from_text(text_response):
    from fpdf import FPDF, XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    lines = text_response.split('\n')
    for line in lines:
        pdf.cell(
            0,
            10,
            text=line.encode('latin-1', 'replace').decode('latin-1'),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )

    # Convert bytearray to bytes
    pdf_bytes = bytes(pdf.output())

    return pdf_bytes

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create the authenticator object
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
)

# Render login form (no need to capture return)
authenticator.login(location='main', max_concurrent_users=5)


def main():
    if st.session_state.get("authentication_status"):
        st.success(f"Welcome {st.session_state['name']} 👋")
        authenticator.logout('Logout', location='sidebar')
        st.sidebar.write(f"Logged in as: {st.session_state['username']}")
        # st.write("Main app content here...")

        # st.title("Synthite AI - QA Specification & Regulation Checker")

        req_file = st.file_uploader("📥 Upload Requirement Sheet PDF", type="pdf")
        pis_file = st.file_uploader("📥 Upload PIS ", type="pdf")

        psm_mode = st.selectbox("🔧 Tesseract OCR PSM", [3, 4, 6, 11], index=2)

        if req_file and pis_file:
            with st.spinner("🔍 Extracting text using OCR..."):
                req_text_raw = ocr_extract_text(req_file, psm=psm_mode)
                pis_text_raw = extract_text(pis_file)

            st.markdown("📝  Requirement Sheet (Cleaned)")
            req_text = emphasize_params(req_text_raw)
            pis_text = emphasize_params(pis_text_raw)

            with st.spinner("🧠 Comparing with Gemini 2.5 Pro..."):
                result_text, total_tokens = compare_documents(req_text, pis_text)

            st.success("✅ Comparison Complete")
            st.markdown("## 📊 Comparison Result")
            st.markdown(result_text)

            # Convert Gemini response table to dataframe
            result_df = parse_markdown_table(result_text)

            # Save Excel
            excel_filename = "comparison_result.xlsx"
            result_df.to_excel(excel_filename, index=False)

            # Save PDF
            pdf_bytes = create_pdf_from_text(result_text)

            # Download buttons
            st.download_button("⬇️ Download Excel Table", data=open(excel_filename, "rb"), file_name=excel_filename)
            st.download_button("⬇️ Download Full Response PDF", data=pdf_bytes, file_name="gemini_response.pdf")

    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")

    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your credentials")


    else:
        st.warning("Awaiting login...")

if __name__ == "__main__":
    main()