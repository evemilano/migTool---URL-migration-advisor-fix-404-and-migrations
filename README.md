# migtool
[Migtool for SEO](https://cluster.army/migtool/)

This tool, the "**404 Error Resolver - URL Redirect Assistant**", is specifically designed to efficiently resolve 404 errors on your website. It assists you by suggesting the most suitable URL redirects, saving you time and effort in the process. The tool automates the identification of similar URLs and proposes potential redirects, streamlining the handling of 404 errors and ensuring a smoother user experience on your website.

# 404 Error Resolver - URL Redirect Assistant

The "404 Error Resolver - URL Redirect Assistant" is a Python-based, sophisticated tool designed to automate the resolution of 404 errors on websites. This tool enhances SEO performance and user experience by streamlining the process of identifying the best possible URL redirects.

## Key Features
- **Automated Identification and Suggestion**: Suggests similar URLs for redirecting 404 errors using advanced string matching techniques.
- **Comprehensive Data Analysis**: Generates a final merged DataFrame, providing a thorough analysis to facilitate informed decision-making.
- **Downloadable Results**: Allows for the downloading of analysis results as an Excel file for convenience and further use.
- For 404 errors, you have the option to upload a [Google Search Console](https://search.google.com/search-console/about) Coverage/404 report or a [Screaming Frog](https://www.screamingfrog.co.uk/seo-spider/) 404 report. This flexibility allows for a comprehensive analysis of broken links from diverse sources. 
- For evaluating potential redirects to live pages, the tool accepts the Google Search Console Coverage/Indexed report or a Screaming Frog report of live, indexable HTML pages. This ensures that your redirects are not only accurate but also beneficial for your site's SEO performance.

## Technologies and Libraries
This script integrates Streamlit for its interactive web application capabilities, along with a suite of powerful Python libraries:
- **Pandas**: For efficient data manipulation and analysis, especially with DataFrame structures.
- **NumPy**: Supports large, multi-dimensional arrays and matrices, alongside a vast collection of mathematical functions.
- **Scipy**: Utilized for `argrelextrema`, which helps in determining local maxima in data, essential for calculating optimal thresholds.
- **Joblib**: Enhances performance through parallel processing.
- **Fuzzywuzzy**: Facilitates fuzzy string comparisons, pivotal in finding URL similarities.
- **Difflib and Levenshtein**: Provides various algorithms for meticulous string matching and comparison.

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

## Usage
This tool is tailored for SEO professionals and website administrators, requiring two types of Excel files: one with 404 URLs and another with live, indexable URLs. Supported sources include Google Search Console and Screaming Frog reports.

## Streamlit Web Interface
Leveraging Streamlit's capabilities, the script offers a user-friendly interface featuring file uploaders, selection boxes, progress bars, and download buttons to enhance user interaction and experience.

## Algorithmic Approach
The tool's effectiveness lies in its diverse algorithmic approach, allowing for a multi-faceted analysis of URLs. Each algorithm brings a unique perspective to the table, contributing to a robust and reliable redirect strategy. The combination of these algorithms under one umbrella tool makes the "404 Error Resolver - URL Redirect Assistant" not just a utility but a comprehensive solution for managing 404 errors effectively.


Begin by uploading your Excel files and let the 404 Error Resolver assist you in resolving those challenging broken links!

**Links:**
- [github](https://github.com/evemilano/migtool)
- [cluster.army](https://cluster.army/)
- [evemilano.com](https://www.evemilano.com)
