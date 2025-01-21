# KiasuKaki - Government Scheme Chatbot and Feedback Analyzer

## Project Description

KiasuKaki is a chatbot application developed for the hackathon challenge: _"How can we leverage Generative AI to enhance or propose new government initiatives and services in smart cities?"_ Our solution utilizes Google's Gemini API to provide citizens with improved access to and understanding of government information within a smart city environment. We offer a conversational interface for complex information, alongside a robust feedback analysis tool to inform future policy and service improvements.

This project tackles the challenge of information accessibility in smart cities, where dense government documents and policies can often overwhelm citizens. By combining natural language processing, efficient document retrieval, and insightful feedback analysis, KiasuKaki empowers users to quickly find clear and concise answers to their questions. This approach promotes a more informed and engaged populace. Additionally, the feedback analysis tool provides valuable, real-time insights that enable government administrators to enhance existing services, and make data driven decisions, improving government efficiency and responsiveness.

**Key Features:**

- **AI-Powered Conversational Chatbot:** Employs Google's Gemini API for natural language understanding and generation, enabling citizens to interact with a chatbot to access government scheme information in an intuitive way.
- **Intelligent Document Retrieval:** Uses ChromaDB for efficient search and retrieval of relevant information from government documents, reducing time spent on manual searches and long documents.
- **Flexible Document Customisation:** Users can easily replace or add to the existing documents in the `/data/documents` folder with updated documents (e.g., Budget 2025 or new schemes). Only run the preprocessing step if you have modified this folder with new or updated documents.
- **Comprehensive Feedback Analysis Tool:** Provides insights into user sentiment and feedback regarding government policies and schemes, enabling data-driven decisions for future improvements.
- **Interactive Dashboard Visualisations:** A dedicated dashboard visualises feedback with charts, graphs and summaries using data-driven approaches.
- **User-Friendly Interface:** Developed with Streamlit to ensure an accessible and intuitive platform for all users, with an easy-to-use and interactive chat interface and dashboard.
- **Data Preprocessing and Storage**: Preprocesses feedback data for the dashboard, for easy access to summarised information without calling the API again, this includes pre-computed AI summarisation.
- **Data Filtering:** The dashboard provides useful data filtering options to make specific analysis easier, such as date filtering and category/sentiment filters.

## Setup Instructions

To set up and run KiasuKaki, follow these steps:

1.  **Clone the Repository:**

    ```
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    It is highly recommended to use a virtual environment to avoid dependency conflicts.

    ```
    python -m venv venv
    ```

    Activate the virtual environment:

    - **On Windows:**

      ```
      venv\Scripts\activate
      ```

    - **On macOS and Linux:**

      ```
      source venv/bin/activate
      ```

3.  **Install Dependencies:**
    Ensure you have Python 3.7+ installed. Then, use pip to install the required packages:

    ```
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, create one in the root of the project directory with this content:

    ```
    streamlit
    python-dotenv
    google-generativeai
    chromadb
    langchain
    pypdf
    pandas
    plotly
    textblob
    ```

4.  **Set up Environment Variables:**

    - Create a `.env` file in the root of the project directory.
    - Add your Google Gemini API key to the `.env` file, for example:

      ```
      GEMINI_API_KEY=YOUR_GEMINI_API_KEY
      ```

    - Obtain your Gemini API key from Google's AI Studio (refer to the Google documentation for details).

    **Note:** As of January 12, 2025, the `gemini-2.0-flash-exp` model is free to use under the Gemini API.

5.  **Document Folder:** The `/data/documents` folder contains the source documents that feed the chatbot.

6.  **Preprocess the document database (if necessary):**

    - Only run the preprocessing step _if you have changed the documents or added new ones_.

      ```
      python preprocessing.py
      ```

    This step processes your PDF documents, creating embeddings using Gemini API, and storing them into the ChromaDB database. Make sure the preprocessing runs without any errors or warnings.

7.  **Run the Chatbot Application:**

    ```
    streamlit run app.py
    ```

    This will start the Streamlit application, accessible via the URL provided in your terminal (usually `http://localhost:8501`).

8.  **Run the Dashboard:**
    ```
    streamlit run dashboard.py
    ```
    This starts the Streamlit dashboard, available at the URL specified in the terminal.

## Usage Guide

### Chatbot Application (`app.py`):

1.  **Access the Application:** Go to the URL provided by Streamlit in your web browser.
2.  **Ask Questions:** Use the chat input at the bottom of the page to ask questions about Singaporean government schemes.
3.  **Receive Responses:** The chatbot will provide relevant and clear answers based _only_ on the information in the documents provided.
4.  **Provide Feedback:** Use the sidebar to submit feedback on the chatbot's performance or the policies and schemes that are being discussed.
5.  **End Session:** Click the button in the sidebar to reset the current chat session and its history.

### Dashboard (`dashboard.py`):

1.  **Access the Dashboard:** Navigate to the provided URL in your browser.
2.  **Preprocess Data:** The first step is to go to the "Preprocess Data" section, select a date range, and press "Process Data" to begin processing the feedback.
3.  **View the Overview Report:** Go to the Overview Report section and select the data range, to see an overview of the feedback such as the average sentiment and summarisation.
4.  **View the Visual Charts:** Use the "Visual Charts" to see the graphical visualisations of the feedback provided, such as a pie chart of sentiment or a bar chart of feedback categories.
5.  **View Specific Feedback:** Use the "View Feedback" page to see all of the feedback data in a table, you can also apply filters such as category and sentiment.
6.  **Modify Settings:** In the "Settings" page, you can adjust the settings for the dashboard.

### Example Queries:

- **Chatbot:** "What are the requirements to be eligible for the cost of living special payment?"
- **Chatbot:** "How much money will I get if I am eligible for the special payment?"

## Contributors

- Koh Jun Sheng
- Matthew Lim Wei Li
- Ben Tan Kiat
- Keith Ng Jun Hao
- Ng Jun Heng, Keith

## Additional Notes

- **Limitations:**
  - The chatbot is limited by the knowledge within the provided document database and does not use external knowledge.
  - The dashboard requires valid feedback data to provide correct analysis.
- **Future Improvements:**
  - Add support for multiple document types beyond PDFs.
  - Implement user authentication.
  - Enhance the chatbot's response format and contextual understanding using advanced Gemini techniques and other LLM tools.
  - Implement a more sophisticated system to extract more granular data from user feedback.
