
import fitz
import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
from fpdf import FPDF, XPos, YPos
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import textwrap
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


from PIL import Image
import pytesseract
import re
import io
from transformers import AutoTokenizer
import google.generativeai as genai

st.set_page_config(page_title="PDF Comparator", layout="wide")
st.title("Synthite AI - QA Specification & Regulation Checker")

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

def convert_markdown_to_excel_hyperlink(text):
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    match = re.match(pattern, text)
    if match:
        display_text = match.group(1)
        url = match.group(2)
        return f'=HYPERLINK("{url}", "{display_text}")'
    return text



default_regulation_links = { '40 CFR (FDA)' :'http://www.ecfr.gov/cgi-bin/text-idx?tpl=/ecfrbrowse/Title40/40tab_02.tpl',
                             'Australian Food Standard' :'https://www.foodstandards.gov.au/',
                             'Codex Alimentarius Updates' :'https://www.fao.org/fao-who-codexalimentarius/en/',
                             'Codex Alimentarius - Pesticide MRLS' :'https://www.fao.org/fao-who-codexalimentarius/codex-texts/dbs/pestres/en/',
                             'European Food Safety Authority' :'http://www.efsa.europa.eu/',
                             'FDA, Thailand': 'http://www.fda.moph.go.th/eng/index.stm',
                             'Food standards code - Australia': 'http://www.foodstandards.gov.au/code/Pages/default.aspx',
                             'Japan MRLs': 'http://www.ffcr.or.jp/zaidan/FFCRHOME.nsf/pages/MRLs-p',
                             'Japan specification & standards' : 'https://www.ffcr.or.jp/en/tenka/',
                             'KFDA (Korea)' : 'https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=72432',
                             'KFDA MRLs'	: 'https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=71065&srchFr=&srchTo=&srchWord=&srchTp=&itm_seq_1=0&itm_seq_2=0&multi_itm_seq=0&company_cd=&company_nm=&page=2',
                             'New Zealand MRLs' : 'https://www.mpi.govt.nz/dmsdocument/19550-Food-Notice-Maximum-Residue-Levels-for-Agricultural-Compounds2025/',
                             'EU MRLs' : 'https://food.ec.europa.eu/plants/pesticides/eu-pesticides-database_en',
                             'FSSAI' : 'http://www.fssai.gov.in/',
                             '21CFR'	: 'https://www.ecfr.gov/current/title-21'}


def load_regulation_mapping(uploaded_excel):
    user_mapping = {}

    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        if "Regulation" in df.columns and "Link" in df.columns:
            user_mapping = dict(zip(df["Regulation"], df["Link"]))

    # Merge default with user-provided, user’s entries overwrite defaults if same key
    combined_mapping = {**default_regulation_links, **user_mapping}
    return combined_mapping


def build_regulation_mapping_prompt(reg_mapping_dict):
    if not reg_mapping_dict:
        return ""

    mapping_text = "| Regulation | Link |\n|------------|------|\n"
    for reg, link in reg_mapping_dict.items():
        mapping_text += f"| {reg} | {link} |\n"

    return (
            "Refer to the regulation mapping below while forming hyperlinks:\n\n"
            + mapping_text
            + "\nUse these mappings wherever applicable."
    )


def compare_documents(req_text, pis_text, reg_mapping_dict):
    req_tokens = count_tokens(req_text)
    pis_tokens = count_tokens(pis_text)
    print(f"\n🔢 Token size of Requirement File: {req_tokens}")
    print(f"🔢 Token size of Product Info File: {pis_tokens}")
    base_prompt = f"""
Compare the following two technical documents as a Quality analyst:

--- Requirement Sheet ---
{req_text}

--- PIS Sheet ---
{pis_text}

Make sure the output includes the table under the heading:

**Detailed Parameter Comparison**

Follow this exact title formatting (including capitalization and bold) so that downstream processing tools can detect and extract this section reliably.


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
10.Check if the unavailable regulations have their respective links is found in the table above where for each regulation we have their respective link.
11.When handling regulations:
    - Do not truncate regulation names — search the **entire regulation phrase** as-is, including commas, editions, dates, and chapter numbers.
    - For example: If the regulation is "BAM, online January, 2001 Chapter: 3", search **this full phrase**.
    - Avoid stopping at partial matches like "BAM".
    - Check if this full regulation phrase is present is found in the table above, and use its respective hyperlink if found.

Output in table format with columns:
| Parameter | Requirement Value | PIS Value | Regulations or Method in Requirement | Regulations or Method in PIS | Match Status (✅ / ⚠️ / ❌) | Reason |
"""

    model = genai.GenerativeModel("gemini-2.5-pro")

    reg_mapping_prompt = build_regulation_mapping_prompt(reg_mapping_dict)


    full_prompt = base_prompt + "\n\n" + reg_mapping_prompt

    response = model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=0.4))
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




def save_text_to_pdf(text, filename="llm_output.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 11)

    x_margin = 1 * inch
    y_margin = 1 * inch
    max_width = width - 2 * x_margin
    y = height - y_margin

    # Wrap text to fit page width
    wrapped_lines = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=100)  # you can tune this
        wrapped_lines.extend(wrapped if wrapped else [""])  # keep blank lines too

    for line in wrapped_lines:
        if y < y_margin:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - y_margin
        c.drawString(x_margin, y, line)
        y -= 14  # line spacing

    c.save()
    with open(filename, "rb") as f:
        return f.read()


def truncate_after_detailed_section(text):
    lines = text.splitlines()
    truncated_lines = []
    for line in lines:
        # Normalize line for matching
        clean_line = line.strip()
        if re.match(r"^(#+\s*)?\*{0,2}\s*Detailed", clean_line, re.IGNORECASE):
            break
        truncated_lines.append(line)
    return "\n".join(truncated_lines)



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
        reg_mapping_file = st.file_uploader("📄 Upload Regulation Mapping Excel", type=["xlsx"])


        psm_mode = st.selectbox("🔧 Tesseract OCR PSM", [3, 4, 6, 11], index=2)
        if "result_text" not in st.session_state:
            if req_file and pis_file:
                with st.spinner("🔍 Extracting text using OCR..."):
                    req_text_raw = ocr_extract_text(req_file, psm=psm_mode)
                    pis_text_raw = extract_text(pis_file)

                st.markdown("📝  Requirement Sheet (Cleaned)")
                req_text = emphasize_params(req_text_raw)
                pis_text = emphasize_params(pis_text_raw)

                # Load regulation mapping
                regulation_mapping_dict = load_regulation_mapping(reg_mapping_file)

                with st.spinner("🧠 Comparing with Gemini 2.5 Pro..."):
                    result_text, total_tokens = compare_documents(req_text, pis_text, reg_mapping_dict=regulation_mapping_dict)

                result_df = parse_markdown_table(result_text)

                for col in ["Regulations or Method in Requirement", "Regulations or Method in PIS"]:
                    if col in result_df.columns:
                        result_df[col] = result_df[col].apply(convert_markdown_to_excel_hyperlink)

                # Save Excel
                excel_filename = "comparison_result.xlsx"
                result_df.to_excel(excel_filename, index=False)

                cleaned_text = truncate_after_detailed_section(result_text)
                pdf_bytes = save_text_to_pdf(cleaned_text, filename="llm_output.pdf")


                # Store in session
                st.session_state["result_text"] = result_text
                st.session_state["result_df"] = result_df
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["excel_filename"] = excel_filename

        # ✅ After session state is populated, show outputs (always!)
        if "result_text" in st.session_state:
            st.success("✅ Comparison Complete")
            st.markdown("## 📊 Comparison Result")
            st.markdown(st.session_state["result_text"])

            st.download_button(
                "⬇️ Download Excel Table",
                data=open(st.session_state["excel_filename"], "rb"),
                file_name=st.session_state["excel_filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.download_button(
                "⬇️ Download Full Response PDF",
                data=st.session_state["pdf_bytes"],
                file_name="gemini_response.pdf",
                mime="application/pdf"
            )
        if st.button("🔄 Reset and Upload New Files"):
            for key in ["result_text", "result_df", "pdf_bytes", "excel_filename"]:
                st.session_state.pop(key, None)


    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")

    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your credentials")


    else:
        st.warning("Awaiting login...")

if __name__ == "__main__":
    main()
