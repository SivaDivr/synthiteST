import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import textwrap
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import re
import os
import io
from transformers import AutoTokenizer

# --- Vertex AI specific imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# --- Page and App Configuration ---
st.set_page_config(page_title="Synthite AI", layout="wide")
st.title("Synthite AI - QA Specification & Regulation Checker")

# --- Environment and Credentials Setup ---
# For local development, this line is fine.
# For deployment, set this as an environment variable in the settings.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'synthiteqa-d1b7d9b06570.json'

# --- Vertex AI Initialization ---
try:
    vertexai.init(project='synthiteqa', location='us-central1')
except Exception as e:
    st.error(f"🚨 Vertex AI Initialization Failed. Ensure your GCP project and location are correct and the credentials file is valid. Error: {e}")
    st.stop()


# --- Helper Functions ---

# Tokenizer for counting output tokens (optional but good for monitoring)
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

def count_tokens(text):
    """Counts the number of tokens in a given text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def convert_markdown_to_excel_hyperlink(text):
    """Converts a markdown link [text](url) to an Excel HYPERLINK formula."""
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    match = re.match(pattern, text)
    if match:
        display_text = match.group(1).replace('"', '""') # Escape double quotes for Excel
        url = match.group(2)
        return f'=HYPERLINK("{url}", "{display_text}")'
    return text

    # If no link is found, return the original text
    return text
# Default regulation links
default_regulation_links = {
    '40 CFR (FDA)': 'http://www.ecfr.gov/cgi-bin/text-idx?tpl=/ecfrbrowse/Title40/40tab_02.tpl',
    'Australian Food Standard': 'https://www.foodstandards.gov.au/',
    'Codex Alimentarius Updates': 'https://www.fao.org/fao-who-codexalimentarius/en/',
    'Codex Alimentarius - Pesticide MRLS': 'https://www.fao.org/fao-who-codexalimentarius/codex-texts/dbs/pestres/en/',
    'European Food Safety Authority': 'http://www.efsa.europa.eu/',
    'FDA, Thailand': 'http://www.fda.moph.go.th/eng/index.stm',
    'Food standards code - Australia': 'http://www.foodstandards.gov.au/code/Pages/default.aspx',
    'Japan MRLs': 'http://www.ffcr.or.jp/zaidan/FFCRHOME.nsf/pages/MRLs-p',
    'Japan specification & standards': 'https://www.ffcr.or.jp/en/tenka/',
    'KFDA (Korea)': 'https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=72432',
    'KFDA MRLs': 'https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=71065&srchFr=&srchTo=&srchWord=&srchTp=&itm_seq_1=0&itm_seq_2=0&multi_itm_seq=0&company_cd=&company_nm=&page=2',
    'New Zealand MRLs': 'https://www.mpi.govt.nz/dmsdocument/19550-Food-Notice-Maximum-Residue-Levels-for-Agricultural-Compounds2025/',
    'EU MRLs': 'https://food.ec.europa.eu/plants/pesticides/eu-pesticides-database_en',
    'FSSAI': 'http://www.fssai.gov.in/',
    '21CFR': 'https://www.ecfr.gov/current/title-21'
}

def load_regulation_mapping(uploaded_excel):
    """Loads user-provided regulation mappings and merges them with defaults."""
    user_mapping = {}
    if uploaded_excel:
        try:
            df = pd.read_excel(uploaded_excel)
            if "Regulation" in df.columns and "Link" in df.columns:
                user_mapping = dict(zip(df["Regulation"], df["Link"]))
        except Exception as e:
            st.warning(f"Could not read the Excel file. Using default regulations. Error: {e}")
    combined_mapping = {**default_regulation_links, **user_mapping}
    return combined_mapping

def build_regulation_mapping_prompt(reg_mapping_dict):
    """Builds a markdown table of regulations for the prompt."""
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

def compare_documents(req_pdf_part, pis_pdf_part, reg_mapping_dict):
    """
    Compares two PDF documents by sending them directly to the Gemini model.
    This version uses the user-specified 11-point prompt exactly as written.
    """
    instructional_prompt = f"""
Analyze and compare the two PDF documents provided: a "Requirement Sheet" and a "PIS Sheet".
Make sure the output includes the table under the heading:

**Detailed Parameter Comparison**

Follow this exact title formatting (including capitalization and bold) so that downstream processing tools can detect and extract this section reliably.



Your primary goal is to produce a markdown table with the heading **Detailed Parameter Comparison**. Follow this title format exactly.

Instructions:
1. List every field from requirement sheet (req_text) in the output table without fail no matter how much the table length.
2. List all parameters that match exactly (value and meaning).
3. List parameters that are semantically similar (e.g., "Aroma" vs "Odour") with values within acceptable limits.
4. List all mismatches (missing, out of range, or incorrect).
5. Identify if both refer to the same or similar meaning, even with different wording.
6. Show extra parameters found in one but not the other.
7. Consider plural words and partially matched words as same.
8. For each regulation you mention (e.g., EU, ASTA, ISO, EC, BAM, AOAC,JECFA,etc.), include a hyperlink to the official regulation page or source if known.
8.1. for sites like ISO dont pick the first site comes unless until its Online Browse Platform (OBP) from iso.org ,for ISO 4883-2 : ISO regulations are listed like "https://www.iso.org/obp/ui/en/#iso:std:iso:4833:-2:ed-1:v1:en" so do like this
8.2. Incase of FCC regulations not the entire FCC site but  try to pick the exact edition as mentioned  for FCC 12th Ed. : https://www.foodchemicalscodex.org/sites/fcconline/files/usp_pdf/EN/fcc/fcc-12-commentary-20200302.pdf take there directly
8.3. for JECFA : https://www.fao.org/food/food-safety-quality/scientific-advice/jecfa/jecfa-additives/detail/en/c/358/
8.4 incase if the extracted regulation is  AOAC regulations not the entire AOAC website but try to pick the exact edition as mentioned for AOAC 19th Ed. 999.11 : https://academic.oup.com/aoac-publications/search-results?f_BookID=45491&fl_SiteID=6535&fl_BookID=45491&cqb=[22terms%22:[%22filter%22:%22%22,%22input%22:%22AOAC%20999.11%22]]&qb=%22q%22:%22AOAC%20999.11%22&page=1
8.5 incase if the extracted regulation is ASTA if Regulations is ASTA 24.2, 1997 or ASTA 24.2 then fetch  https://astaspice.org/resources/search?q=ASTA%2024.2 like this
9. If the exact link is unavailable, suggest the most likely official website or portal where the regulation can be found ( europa.eu for EU regulations, etc.).
10. Check if the unavailable regulations have their respective links is found in the table above where for each regulation we have their respective link.
11. When handling regulations:
    - Do not truncate regulation names — search the **entire regulation phrase** as-is, including commas, editions, dates, and chapter numbers.
    - For example: If the regulation is "BAM, online January, 2001 Chapter: 3", search **this full phrase**.
    - Avoid stopping at partial matches like "BAM".
    - Check if this full regulation phrase is present is found in the table above, and use its respective hyperlink if found.

Output in table format with columns:
| Parameter | Requirement Value | PIS Value | Regulations or Method in Requirement | Regulations or Method in PIS | Match Status (✅ / ⚠️ / ❌) | Reason |
"""
    reg_mapping_prompt = build_regulation_mapping_prompt(reg_mapping_dict)
    full_prompt = instructional_prompt + "\n\n" + reg_mapping_prompt

    # Use a model that supports native PDF reading, like Gemini 1.5 Pro.
    model = GenerativeModel("gemini-2.5-pro")

    # Send the prompt text and the two PDF 'Part' objects in a single list
    response = model.generate_content(
        [
            req_pdf_part,
            pis_pdf_part,
            full_prompt
        ],
        generation_config=GenerationConfig(temperature=0.4)
    )

    result_text = response.text.strip()
    response_tokens = count_tokens(result_text)
    print(f"🧮 Response Tokens: {response_tokens}")

    return result_text, response_tokens

# --- Parsing and File Saving Functions ---

def parse_markdown_table(markdown_text):
    """Parses a markdown table from the model's response into a DataFrame."""
    # Find the start of the detailed comparison table
    table_start_marker = "**Detailed Parameter Comparison**"
    if table_start_marker not in markdown_text:
        st.warning("⚠️ The specific 'Detailed Parameter Comparison' table was not found in the response.")
        return pd.DataFrame()

    table_text = markdown_text.split(table_start_marker, 1)[1]
    lines = table_text.strip().split('\n')
    table_lines = [line for line in lines if line.strip().startswith('|')]

    if len(table_lines) < 2: # Header + Separator
        st.warning("⚠️ No valid table data found after the heading.")
        return pd.DataFrame()

    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip('|').split('|')]
    expected_cols = len(headers)

    valid_rows = []
    for line in table_lines[2:]: # Skip header and separator lines
        row = [cell.strip() for cell in line.strip('|').split('|')]
        if len(row) == expected_cols:
            valid_rows.append(row)

    if not valid_rows:
        st.warning("⚠️ Table headers were found, but no data rows could be parsed.")
        return pd.DataFrame()

    df = pd.DataFrame(valid_rows, columns=headers)
    return df

def save_text_to_pdf(text, filename="llm_output.pdf"):
    """Saves a string of text to a PDF file."""
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 10)
    x_margin, y_margin = 1 * inch, 1 * inch
    max_width = width - 2 * x_margin
    y = height - y_margin

    wrapped_lines = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=100) # Adjust wrap width as needed
        wrapped_lines.extend(wrapped if wrapped else [""])

    for line in wrapped_lines:
        if y < y_margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - y_margin
        c.drawString(x_margin, y, line)
        y -= 12 # Line spacing

    c.save()
    with open(filename, "rb") as f:
        return f.read()

def truncate_before_detailed_section(text):
    """Extracts only the text that comes *before* the main comparison table."""
    match = re.search(r"(\*\*Detailed Parameter Comparison\*\*)", text, re.IGNORECASE)
    if match:
        return text[:match.start()]
    return text # Return full text if marker not found


# --- Authentication and Main App Logic ---

def main():
    # Load config from YAML file
    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error("🚨 `config.yaml` not found. Please create it for authentication.")
        st.stop()

    # Create the authenticator object
    authenticator = stauth.Authenticate(
        credentials=config['credentials'],
        cookie_name=config['cookie']['name'],
        cookie_key=config['cookie']['key'],
        cookie_expiry_days=config['cookie']['expiry_days'],
    )

    # Render login form
    authenticator.login(location='main', max_concurrent_users=5)

    # AUTHENTICATION: The main app is now wrapped in a clean if/elif/else block
    # to handle all authentication states (logged in, failed, not logged in).
    if st.session_state.get("authentication_status"):
        st.sidebar.success(f"Welcome {st.session_state['name']} 👋")
        authenticator.logout('Logout', location='sidebar')

        # --- Main App Interface ---
        req_file = st.file_uploader("📥 Upload Requirement Sheet PDF", type="pdf")
        pis_file = st.file_uploader("📥 Upload PIS (Product Information Sheet) PDF", type="pdf")
        reg_mapping_file = st.file_uploader("📄 (Optional) Upload Custom Regulation Mapping", type=["xlsx"])

        # This block contains the processing logic, which runs only on button click
        if st.button("🚀 Compare Documents", disabled=(not req_file or not pis_file)):
            with st.spinner("🧠 Analyzing documents with Gemini... This may take a moment."):
                # Get PDF data as Part objects for the model
                req_pdf_part = Part.from_data(req_file.getvalue(), mime_type="application/pdf")
                pis_pdf_part = Part.from_data(pis_file.getvalue(), mime_type="application/pdf")

                regulation_mapping_dict = load_regulation_mapping(reg_mapping_file)

                # Call the AI comparison function
                result_text, _ = compare_documents(
                    req_pdf_part,
                    pis_pdf_part,
                    reg_mapping_dict=regulation_mapping_dict
                )

                result_df = parse_markdown_table(result_text)

                # Post-process DataFrame for Excel hyperlinks
                for col in ["Regulations or Method in Requirement", "Regulations or Method in PIS"]:
                    if col in result_df.columns:
                        result_df[col] = result_df[col].apply(convert_markdown_to_excel_hyperlink)

                # Prepare files in memory for download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False)
                excel_bytes = excel_buffer.getvalue()

                summary_text = truncate_before_detailed_section(result_text)
                pdf_bytes = save_text_to_pdf(summary_text, filename="llm_summary.pdf")

                # Store all generated artifacts in the session state
                st.session_state["result_text"] = result_text
                st.session_state["excel_bytes"] = excel_bytes
                st.session_state["pdf_bytes"] = pdf_bytes

        # This block displays results if they exist in the session state
        if "result_text" in st.session_state:
            st.success("✅ Comparison Complete")
            st.markdown("---")
            st.markdown("## 📝 Gemini's Full Response")
            st.markdown(st.session_state["result_text"])
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Comparison Table (Excel)",
                    data=st.session_state["excel_bytes"],
                    file_name="comparison_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "⬇️ Download Summary (PDF)",
                    data=st.session_state["pdf_bytes"],
                    file_name="comparison_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # RESETTING: A dedicated "Reset" button to clear results and start over easily.
        if st.sidebar.button("🔄 Reset and Start Over"):
            # A list of keys to remove from the session state
            keys_to_clear = ["result_text", "result_df", "excel_bytes", "pdf_bytes"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Rerun the app to refresh the interface
            st.rerun()

    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your credentials to use the application.")

# This line ensures the main function runs when the script is executed
if __name__ == "__main__":
    main()