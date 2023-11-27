import streamlit as st
import pandas as pd
import time
from scipy.signal import argrelextrema
import numpy as np
import io
from urllib.parse import urlparse, unquote
import re
from fuzzywuzzy import fuzz
import difflib
import Levenshtein as lev
from joblib import Parallel, delayed

##########################################################################################################


@st.cache(allow_output_mutation=True, show_spinner=True)
def load_excel(uploaded_file):
    print("Loading Excel file...")  # Debug print
    if uploaded_file is not None:
        print(f"Uploaded file: {uploaded_file.name}")  # Print the file name
        if uploaded_file.name.endswith('.xlsx'):  # Verifica che sia un file Excel
            try:
                # Carica il file Excel con tutti i fogli
                sheets = pd.read_excel(uploaded_file, sheet_name=None)
                print("Excel file loaded successfully.")  # Debug print
                return sheets, ""  # Restituisce i fogli e un messaggio di errore vuoto
            except Exception as e:
                print(f"Error loading Excel file: {e}")  # Print the error
                return None, f"Errore: {e}"  # Restituisce None e il messaggio di errore
        else:
            print("Uploaded file is not an Excel file.")  # Debug print
            return None, "Il file caricato non è un file Excel (.xlsx)."  # Restituisce None e il messaggio di errore
    else:
        print("No file uploaded.")  # Debug print
        return None, "Nessun file caricato."  # Restituisce None e il messaggio di errore

def show_excel_column_headers(sheets, selected_sheet):
    if selected_sheet in sheets:
        return sheets[selected_sheet].columns.tolist()
    return []


def is_file_size_within_limit(uploaded_file, max_size_mb=50):
    if uploaded_file is not None:
        # Convert max size to bytes
        max_size_bytes = max_size_mb * 1024 * 1024
        if uploaded_file.size > max_size_bytes:
            return False
    return True


def clean_url(url):
    # Parse the URL
    parsed = urlparse(unquote(url))
    path = ' '.join(parsed.path.split('/'))
    params = parsed.params
    if params:
        params = '&'.join(sorted(params.split('&')))
    path = re.sub(r'[-_#]', ' ', path)
    return path + ('?' + params if params else '')


def find_best_match_fuzzy(url, live_df, selected_column, threshold):
    cleaned_url = clean_url(url)
    best_score = 0
    best_match = None
    for index, row in live_df.iterrows():
        score = fuzz.token_sort_ratio(cleaned_url, row['Cleaned URLs'])
        if score > best_score:
            best_score = score
            best_match = row[selected_column]

    if best_score >= threshold:
        return best_match
    return None

# Trova la soglia ottimale utilizzando l'algoritmo dell'"elbow" come nell'esempio 2
def find_optimal_threshold(max_similarity_scores):
    hist, bin_edges = np.histogram(max_similarity_scores, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_derivative = np.diff(hist) / np.diff(bin_centers)
    extrema = argrelextrema(hist_derivative, np.greater)
    if extrema[0].size > 0:
        elbow_index = extrema[0][0]
        threshold = bin_centers[elbow_index]
    else:
        print("No extrema found. Falling back to default threshold of 50.")
        threshold = 50  # Imposta una soglia di default se non vengono trovati estremi.
    return threshold

# funzione try 2 più veloce
def find_most_similar_levenshtein(url, live_df, selected_column):
    cleaned_url = clean_url(url)
    # Apply the Levenshtein distance calculation across the 'Cleaned URLs' column
    distances = live_df['Cleaned URLs'].apply(lambda x: lev.distance(cleaned_url, x))
    # Find the index of the minimum distance
    min_index = distances.idxmin()
    # Return the corresponding value from the selected column
    return live_df.at[min_index, selected_column]


def compute_max_similarity(live_url, mig_df_404):
    max_score = 0
    for dev_url in mig_df_404['Cleaned URLs']:
        score = fuzz.token_sort_ratio(live_url, dev_url)
        max_score = max(max_score, score)
    return max_score
    

def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def find_most_similar_jaccard(url, live_df):
    cleaned_url = clean_url(url)
    best_score = 0
    best_match = None
    for index, row in live_df.iterrows():
        score = jaccard_similarity(cleaned_url, row['Cleaned URLs'])
        if score > best_score:
            best_score = score
            best_match = row['URL']
    return best_match, best_score


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Le stringhe devono avere la stessa lunghezza")
    return sum(el1 != el2 for el1, el2 in zip(str1, str2))
    
def find_most_similar_hamming(url, live_df):
    cleaned_url = clean_url(url)
    best_score = float('inf')
    best_match = None
    for index, row in live_df.iterrows():
        if len(cleaned_url) == len(row['Cleaned URLs']):
            score = hamming_distance(cleaned_url, row['Cleaned URLs'])
            if score < best_score:
                best_score = score
                best_match = row['URL']
    return best_match, best_score

def ratcliff_obershelp_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()
    
def find_most_similar_ratcliff(url, live_df):
    cleaned_url = clean_url(url)
    best_score = 0
    best_match = None
    for index, row in live_df.iterrows():
        score = ratcliff_obershelp_similarity(cleaned_url, row['Cleaned URLs'])
        if score > best_score:
            best_score = score
            best_match = row['URL']
    return best_match, best_score

def tversky_index(str1, str2, alpha=0.5, beta=0.5):
    set1 = set(str1)
    set2 = set(str2)
    common = set1.intersection(set2)
    unique_to_set1 = set1 - set2
    unique_to_set2 = set2 - set1
    score = len(common) / (len(common) + alpha * len(unique_to_set1) + beta * len(unique_to_set2))
    return score
    
def find_most_similar_tversky(url, live_df):
    cleaned_url = clean_url(url)
    best_score = 0
    best_match = None
    for index, row in live_df.iterrows():
        score = tversky_index(cleaned_url, row['Cleaned URLs'])
        if score > best_score:
            best_score = score
            best_match = row['URL']
    return best_match, best_score

def count_agreement(row, selected_algorithms):
    # Estrai gli URL suggeriti da ciascun algoritmo, escludendo i valori NA
    suggested_urls = [row[algo] for algo in selected_algorithms if pd.notna(row[algo])]
    # Se non ci sono URL validi suggeriti, restituisci 0 e None
    if not suggested_urls:
        return 0, None
    # Conta il numero di volte che ciascun URL appare
    url_counts = {url: suggested_urls.count(url) for url in set(suggested_urls)}
    # Trova l'URL con il conteggio massimo
    best_redirect = max(url_counts, key=url_counts.get)
    # Restituisce il numero massimo di algoritmi in accordo su un URL e l'URL stesso
    max_count = url_counts[best_redirect]
    return max_count, best_redirect
    
# Aggiornamento del dataframe per includere la colonna 'Agreement' e 'Best redirect'
def update_dataframe_with_agreement_and_best_redirect(df, selected_algorithms):
    # Applica la funzione count_agreement e crea due nuove colonne
    agreements_best_redirects = df.apply(lambda row: count_agreement(row, selected_algorithms), axis=1)
    df['Agreement'], df['Best redirect'] = zip(*agreements_best_redirects)
    
    
# Funzione per convertire il DataFrame in Excel
def convert_df_to_excel(df):
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
    return excel_buffer.getvalue()
                
################################################################################################################################


# Disabilita il badge "Made with Streamlit"
st.set_page_config(layout="wide", page_title="404 Error Resolver - URL Redirect Assistant", page_icon=":tada:", initial_sidebar_state="expanded", menu_items={
    'Get Help': 'https://cluster.army',
    'About': "# This is a free SEO tool made by Giovanni Sacheli."
})


def main():
    
    # rimuovi hamburger e firma footer
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden !important;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("404 Error Resolver - URL Redirect Assistant")
    # Descrizione delle funzionalità del tool
    with st.expander("Click here to expand/hide text", expanded=False):


        st.markdown("""
        ## Key Features
        This tool, the "**404 Error Resolver - URL Redirect Assistant**", is specifically designed to efficiently resolve 404 errors on your website. It assists you by suggesting the most suitable URL redirects, saving you time and effort in the process. The tool automates the identification of similar URLs and proposes potential redirects, streamlining the handling of 404 errors and ensuring a smoother user experience on your website.
        The "404 Error Resolver - URL Redirect Assistant" is a Python-based, sophisticated tool designed to automate the resolution of 404 errors on websites. This tool enhances SEO performance and user experience by streamlining the process of identifying the best possible URL redirects.
        
        - **Automated Identification and Suggestion**: Suggests similar URLs for redirecting 404 errors using advanced string matching techniques.
        - **Comprehensive Data Analysis**: Generates a final merged DataFrame, providing a thorough analysis to facilitate informed decision-making.
        - **Downloadable Results**: Allows for the downloading of analysis results as an Excel file for convenience and further use.
        - For 404 errors, you have the option to upload a [Google Search Console](https://search.google.com/search-console/about) Coverage/404 report or a [Screaming Frog](https://www.screamingfrog.co.uk/seo-spider/) 404 report. This flexibility allows for a comprehensive analysis of broken links from diverse sources. 
        - For evaluating potential redirects to live pages, the tool accepts the Google Search Console Coverage/Indexed report or a Screaming Frog report of live, indexable HTML pages. This ensures that your redirects are not only accurate but also beneficial for your site's SEO performance.

        ## Workflow and Logic
        The script orchestrates the following workflow:
        1. **File Upload and Size Validation**: Allows users to upload Excel files containing URLs, validating against a predefined file size limit.
        2. **Data Loading and Cleaning**: Uploaded files are processed into Pandas DataFrames. URLs undergo a cleaning and standardization process for uniform analysis.
        3. **Algorithm Selection**: Users select from several string matching algorithms, including:
           - **Fuzzy Matching**: For approximate string comparisons.
           - **Levenshtein Distance**: Measures the difference between sequences.
           - **Jaccard Similarity**: Assesses the similarity and diversity of sample sets.
           - **Hamming Distance**: Calculates the distance between string sequences.
           - **Ratcliff/Obershelp**: Provides a ratio of the sequences' similarity.
           - **Tversky Index**: A generalized form of set similarity.
        4. **Similarity Calculation and Threshold Determination**: 
           - Parallel processing is used for swift calculation of similarity scores.
           - The "elbow" method identifies the optimal threshold for similarity.
        5. **Agreement Counting and DataFrame Update**: 
           - Counts algorithm agreement on URL redirects, updating the DataFrame with these insights.
           - This crucial step highlights the most consistent redirect URL across different algorithms.
        6. **Result Presentation and Download**: 
           - Displays the final merged DataFrame with suggested redirects and algorithm agreements.
           - Provides an option to download this data as an Excel file.

        ## Technologies and Libraries
        This script integrates Streamlit for its interactive web application capabilities, along with a suite of powerful Python libraries:
        - **Pandas**: For efficient data manipulation and analysis, especially with DataFrame structures.
        - **NumPy**: Supports large, multi-dimensional arrays and matrices, alongside a vast collection of mathematical functions.
        - **Scipy**: Utilized for `argrelextrema`, which helps in determining local maxima in data, essential for calculating optimal thresholds.
        - **Joblib**: Enhances performance through parallel processing.
        - **Fuzzywuzzy**: Facilitates fuzzy string comparisons, pivotal in finding URL similarities.
        - **Difflib and Levenshtein**: Provides various algorithms for meticulous string matching and comparison.

        ## Usage
        This tool is tailored for SEO professionals and website administrators, requiring two types of Excel files: one with 404 URLs and another with live, indexable URLs. Supported sources include Google Search Console and Screaming Frog reports.

        ## Streamlit Web Interface
        Leveraging Streamlit's capabilities, the script offers a user-friendly interface featuring file uploaders, selection boxes, progress bars, and download buttons to enhance user interaction and experience.

        ## Algorithmic Approach
        The tool's effectiveness lies in its diverse algorithmic approach, allowing for a multi-faceted analysis of URLs. Each algorithm brings a unique perspective to the table, contributing to a robust and reliable redirect strategy. The combination of these algorithms under one umbrella tool makes the "404 Error Resolver - URL Redirect Assistant" not just a utility but a comprehensive solution for managing 404 errors effectively.

        Begin by uploading your Excel files and let the 404 Error Resolver assist you in resolving those challenging broken links!

        ## Links
        - [github](https://github.com/evemilano/migtool)
        - [cluster.army](https://cluster.army/)
        - [evemilano.com](https://www.evemilano.com)

        """)


    col1, col2 = st.columns(2)
    mig_df_404 = None
    mig_df_live = None

    # Memorizza la selezione delle colonne
    selected_column_1 = None
    selected_column_2 = None

    with col1:
        st.header("404 URLs")
        uploaded_file_1 = st.file_uploader("Upload an Excel file here with your 404 error URLs. You can upload the Google Search Console [Coverage/404] export or a Screaming Frog 404 errors report.", type=['xlsx'], key="file_uploader_1")
        if uploaded_file_1 and not is_file_size_within_limit(uploaded_file_1):
            st.error("The file is too large. Please upload a file smaller than 50 MB.")
            uploaded_file_1 = None  # Reset the uploaded file
        elif uploaded_file_1:
            sheets_1, error_1 = load_excel(uploaded_file_1)
            if sheets_1:
                selected_sheet_1 = st.selectbox("Select the Excel sheet.", list(sheets_1.keys()), key="sheet_selector_1")
                column_headers_1 = show_excel_column_headers(sheets_1, selected_sheet_1)
                selected_column_1 = st.selectbox("Select the columns with 404 URLs.", column_headers_1, key="column_selector_1")
            else:
                st.write(error_1)

    with col2:
        st.header("Live URLs")
        uploaded_file_2 = st.file_uploader("Upload an Excel file here with your live and indexable URLs. You can upload the Google Search Console [Covergage/indexed] export or a Screaming Frog All/HTML indexable pages report.", type=['xlsx'], key="file_uploader_2")
        if uploaded_file_2 and not is_file_size_within_limit(uploaded_file_2):
            st.error("The file is too large. Please upload a file smaller than 50 MB.")
            uploaded_file_2 = None  # Reset the uploaded file
        elif uploaded_file_2:
            sheets_2, error_2 = load_excel(uploaded_file_2)
            if sheets_2:
                selected_sheet_2 = st.selectbox("Select the Excel sheet.", list(sheets_2.keys()), key="sheet_selector_2")
                column_headers_2 = show_excel_column_headers(sheets_2, selected_sheet_2)
                selected_column_2 = st.selectbox("Select the columns with Live URLs.", column_headers_2, key="column_selector_2")
            else:
                st.write(error_2)

    # Pulsante per generare i dataframe
    if uploaded_file_1 and uploaded_file_2 and selected_column_1 and selected_column_2:
    
        # Parte del codice dove vengono presentati gli algoritmi all'utente per la selezione
        
        st.write("Select algorithms to use:")
        use_fuzzy = st.checkbox("Use Fuzzy Matching", value=True)
        use_levenshtein = st.checkbox("Use Levenshtein Distance", value=True)
        use_jaccard = st.checkbox("Use Jaccard Similarity", value=False)
        use_hamming = st.checkbox("Use Hamming Distance", value=False)
        use_ratcliff = st.checkbox("Use Ratcliff/Obershelp", value=False)
        use_tversky = st.checkbox("Use Tversky Index", value=False)





        # Controlla se almeno un algoritmo è stato selezionato
        at_least_one_algorithm_selected = use_fuzzy or use_levenshtein or use_jaccard or use_hamming or use_ratcliff or use_tversky

        # Mostra il pulsante solo se almeno un algoritmo è stato selezionato e sono stati caricati entrambi i file
        if at_least_one_algorithm_selected and uploaded_file_1 and uploaded_file_2 and selected_column_1 and selected_column_2:

            if st.button('Genera Dataframes'):
                # Prima carica i DataFrame
                mig_df_404 = pd.DataFrame(sheets_1[selected_sheet_1][selected_column_1])
                mig_df_live = pd.DataFrame(sheets_2[selected_sheet_2][selected_column_2])

                # Verifica che la colonna 'URL' sia presente
                if selected_column_1 not in sheets_1[selected_sheet_1].columns:
                    st.error(f"Column '{selected_column_1}' not found in the uploaded file.")
                    return  # Interrompe l'esecuzione ulteriore del codice

                # Rinomina la colonna selezionata in 'URL'
                mig_df_404.rename(columns={selected_column_1: 'URL'}, inplace=True)

                # Pulizia dei dati in ciascun dataframe
                
                mig_df_404['Cleaned URLs'] = mig_df_404['URL'].apply(clean_url)
                mig_df_live['Cleaned URLs'] = mig_df_live[selected_column_2].apply(clean_url)
                # rinomina la prima colonna
                mig_df_404.columns.values[0] = 'URL'
                mig_df_live.columns.values[0] = 'URL'
                # rimuovi duplicati
                st.write("Removing duplicates...")  # Message above the progress bar
                mig_df_404 = mig_df_404.drop_duplicates(subset='URL')
                mig_df_live = mig_df_live.drop_duplicates(subset='URL')

                # Stampa solo le prime 10 righe dei dataframe
                st.write("Top records - 404 URLs:")
                st.table(mig_df_404.head(3))
                #st.write(f"Total rows in 404 dataframe: {len(mig_df_404)}")  # Aggiunto
                st.write(f"Total rows in 404 dataframe: {len(mig_df_404):,}")  # Con formattazione delle migliaia

                st.write("Top records - Live URLs:")
                st.table(mig_df_live.head(3))
                #st.write(f"Total rows in Live dataframe: {len(mig_df_live)}")  # Aggiunto
                st.write(f"Total rows in Live dataframe: {len(mig_df_live):,}")  # Con formattazione delle migliaia

                # genera df finale
                final_mig_df = mig_df_404.copy()

                # Dopo aver generato il dataframe finale
                selected_algorithms = []
                if use_fuzzy:
                    selected_algorithms.append('Fuzzy')
                if use_levenshtein:
                    selected_algorithms.append('Levenshtein')
                if use_jaccard:
                    selected_algorithms.append('Jaccard')
                if use_hamming:
                    selected_algorithms.append('Hamming')
                if use_ratcliff:
                    selected_algorithms.append('Ratcliff')
                if use_tversky:
                    selected_algorithms.append('Tversky')

                # Assumendo che 'selected_algorithms' sia una lista degli algoritmi selezionati
                total_algorithms = len(selected_algorithms)
                total_iterations_per_algorithm = len(mig_df_live) * len(mig_df_404)
                total_iterations = total_iterations_per_algorithm * total_algorithms
                # Calcolo del tempo medio per iterazione (in minuti)
                tempo_medio_per_iterazione = 23 / 1004772  # 23 minuti diviso per il numero totale di iterazioni
                # Calcolo del tempo stimato per tutte le iterazioni (in minuti)
                tempo_stimato_totale = tempo_medio_per_iterazione * total_iterations
                st.write(f"Total iterations for all selected algorithms: {total_iterations:,}. ETA: {tempo_stimato_totale:.2f} min")

                # Inizia la barra di progresso
                my_bar = st.progress(0)    
                # All'inizio del tuo script o all'interno della funzione principale
                placeholder = st.empty()                
                placeholder.write("Similarity calculation...")  # Message above the progress bar
                
                if use_fuzzy:
                    placeholder.write("Fuzzy...")  # Message above the progress bar
                    # Calcola matrice di similarità senza la stampa del progresso 
                    # Parallel computation of max similarity scores
                    num_cores = -1  # Use all available cores
                    # Calcolo e stampa del numero totale di combinazioni da calcolare
                    total_combinations = len(mig_df_live) * len(mig_df_404)
                    max_similarity_scores = Parallel(n_jobs=num_cores)(delayed(compute_max_similarity)(live_url, mig_df_404) for live_url in mig_df_live['Cleaned URLs'])
                    #st.write("Optimal threshold calculation...")  # Message above the progress bar
                    # Find optimal threshold
                    threshold = find_optimal_threshold(np.array(max_similarity_scores))
                    # FUZZY Calcola la soglia ottimale
                    final_mig_df['Fuzzy'] = final_mig_df['URL'].apply(lambda url: find_best_match_fuzzy(url, mig_df_live, "URL", threshold))
                    # Calcola il punteggio Fuzzy e aggiungilo alla colonna "Fuzzy" nel DataFrame finale
                    final_mig_df['Fuzzy_Score'] = final_mig_df['URL'].apply(lambda url: fuzz.token_sort_ratio(clean_url(url), clean_url(final_mig_df['Fuzzy'].iloc[0])))
                    # print di verifica
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(20)
                    # Levenshtein - Aggiungi i punteggi di similarità alle colonne "Fuzzy" e "Levenshtein"
                
                if use_levenshtein:
                    placeholder.write("Levenshtein...")  # Message above the progress bar
                    final_mig_df['Levenshtein'] = final_mig_df['URL'].apply(lambda url: find_most_similar_levenshtein(url, mig_df_live, "URL"))
                    final_mig_df['Levenshtein_Score'] = final_mig_df['URL'].apply(lambda url: lev.distance(clean_url(url), clean_url(final_mig_df['Levenshtein'].iloc[0])))
                    # print di verifica
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(40)

                if use_jaccard:
                    placeholder.write("Jaccard...")
                    # Applica Jaccard e aggiungi i risultati al DataFrame finale
                    final_mig_df[['Jaccard', 'Jaccard_Score']] = final_mig_df['URL'].apply(lambda url: find_most_similar_jaccard(url, mig_df_live)).apply(pd.Series)
                    # print di verifica
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(50)
                if use_hamming:
                    placeholder.write("Hamming...")
                    # Applica Hamming e aggiungi i risultati al DataFrame finale
                    final_mig_df[['Hamming', 'Hamming_Score']] = final_mig_df['URL'].apply(lambda url: find_most_similar_hamming(url, mig_df_live)).apply(pd.Series)
                    # print di verifica
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(60)

                if use_ratcliff:
                    placeholder.write("Ratcliff...")
                    final_mig_df[['Ratcliff', 'Ratcliff_Score']] = final_mig_df['URL'].apply(lambda url: find_most_similar_ratcliff(url, mig_df_live)).apply(pd.Series)
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(70)
                    
                if use_tversky:
                    placeholder.write("Tversky...")
                    final_mig_df[['Tversky', 'Tversky_Score']] = final_mig_df['URL'].apply(lambda url: find_most_similar_tversky(url, mig_df_live)).apply(pd.Series)
                    # print di verifica
                    #st.table(final_mig_df.head(2))
                    my_bar.progress(80)

                my_bar.progress(90)
                # Mostra il dataframe finale
                placeholder.write("Final Merged Dataframe")

                # Solo se almeno un algoritmo è stato selezionato
                placeholder.write("Check agreements")
                # Aggiungi la colonna 'agreement'
                final_mig_df['Agreement'] = final_mig_df.apply(lambda row: count_agreement(row, selected_algorithms), axis=1)            
                # Chiamata della funzione per aggiornare il dataframe
                update_dataframe_with_agreement_and_best_redirect(final_mig_df, selected_algorithms)
                my_bar.progress(95)

                # rimuovi colonna Cleaned
                final_mig_df.drop(columns=['Cleaned URLs'], inplace=True)

                # Stampa il dataframe finale con la nuova colonna 'Agreement'
                placeholder.write("Process ended!")
                my_bar.progress(100)
                
                # Stampa del dataframe finale con la nuova colonna 'Best redirect'
                final_mig_df.columns.values[0] = '404 URL'
                st.table(final_mig_df.head(10))            

                # Crea il pulsante di download per il file Excel
                excel = convert_df_to_excel(final_mig_df)
                st.download_button(
                    label="Download Excel",
                    data=excel,
                    file_name="final_mig_df.xlsx",
                    mime="application/vnd.ms-excel",
                )

if __name__ == "__main__":
    main()