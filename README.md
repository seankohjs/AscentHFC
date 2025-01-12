# KiasuKaki - Government Scheme Chatbot and Feedback Analyser

## Project Description

KiasuKaki is a chatbot application developed in response to the hackathon challenge: _How can we leverage Generative AI to enhance or propose new government initiatives and services in smart cities?_ Our solution harnesses the power of Generative AI through Google's Gemini API to improve access to and understanding of government information for citizens within a smart city context. We aim to enhance existing services by providing a conversational interface to complex information, alongside a tool for feedback analysis to further shape future initiatives.

This project addresses the challenge of information accessibility in smart cities, where complex government documents and initiatives can be overwhelming for citizens. By combining natural language processing with efficient document retrieval and analysis, KiasuKaki allows users to easily obtain clear and concise answers to their questions, promoting a more informed and engaged populace. Furthermore, the included feedback analysis tool enables city administrators to use real-time feedback to improve current services, which enhances government efficiency and responsiveness.

**Key Features:**

- **AI-Powered Chatbot:** Utilises Google's Gemini API for natural language understanding and generation, providing citizens with an intuitive way to access information about government schemes.
- **Intelligent Document Retrieval:** Leverages ChromaDB to efficiently search and retrieve relevant information from government documents, reducing time spent searching for specific answers.
- **Contextual Responses:** Generates responses based on the provided documents, ensuring information is accurate and contextually relevant, whilst avoiding external knowledge or assumptions.
- **Pre-Processed Document Database:** The ChromaDB database has been pre-processed with documents from Budget 2024, available in the `documents` folder.
- **Document Customisation:** Users are free to replace or add to the existing documents in the `documents` folder with updated ones (e.g., Budget 2025). Preprocessing is only required when new documents are added or existing documents are updated.
- **Feedback Analysis Tool:** Provides insights into public sentiment regarding government schemes, enabling data-driven decisions for improvement.
- **User-Friendly Interface:** Built with Streamlit to offer an accessible and easy-to-use platform for all citizens.


## Setup Instructions

To get the KiasuKaki project up and running, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to create a virtual environment to avoid conflicts with other Python projects.

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
    Make sure you have Python 3.7+ installed. Then, install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, create one in the root of the project directory with the following content:
    `streamlit
 python-dotenv
 google-generativeai
 chromadb
 langchain
 pypdf
`

4.  **Set up Environment Variables:**

    - Create a `.env` file in the root of the project directory.
    - Add your Google Gemini API key to the `.env` file:
      ```
      GEMINI_API_KEY=YOUR_GEMINI_API_KEY
      ```
  
    - You'll need to obtain a Gemini API key from Google's AI Studio, refer to the Google documentation for more information on how to do this.
        **Note:** As of 12 Jan 2025, the `gemini-2.0-flash-exp` model is free to use under the Gemini API, so there is no need to pay for usage.

5.  **Document Folder:** The `documents` folder contains the current Budget 2024 documents that are fed to ChromaDB.

6.  **Run the preprocessing script (if necessary):**

- **Only if you have changed the documents or added new ones**, run the following code

  ```
  python actualpreprocessing.py
  ```

  This step preprocesses your PDF documents, chunks them into smaller pieces, creates embeddings using the Gemini API, and stores them into a ChromaDB database. Ensure this runs without any errors or warnings.

7.  **Run the Chatbot Application:**

    ```
    streamlit run app.py
    ```

    This command starts the Streamlit application, and you can access it in your browser.

8.  **Run the Feedback Analysis Tool:**
    ```
     streamlit run feedback.py
    ```
    This command starts the Streamlit feedback analysis tool, you can access it in your browser.

## Usage Guide

### Chatbot Application (app.py):

1.  **Access the Application:** Open your web browser and go to the URL provided by Streamlit after running the `streamlit run app.py` command (usually `http://localhost:8501`).
2.  **Ask Questions:** Type your questions about government schemes in the chat input box at the bottom of the page.
3.  **Read Responses:** The chatbot will respond with relevant information based on the documents provided.
4.  **Provide Feedback:** Use the sidebar to submit feedback on the chatbot's performance or the schemes in general.
5.  **End Chat Session:** You can end the current chat session with the button in the sidebar, this will clear the session and the history, as well as the token counts.

### Feedback Analysis Tool (feedback.py):

1. **Access the Tool:** Open your web browser and go to the URL provided by Streamlit after running `streamlit run feedback.py` (usually `http://localhost:8501`).
2. **View Raw Feedback:** Expand the "View Raw Feedback" section to see the feedback that users have submitted through the chatbot.
3. **Ask Questions:** Type your specific questions related to feedback analysis in the chat input box.
4. **Receive Analysis:** The chatbot will respond based on the data available, you can ask questions like the average rating, or for a summarisation of the feedback given
5. **End Chat Session:** You can end the current chat session with the button in the sidebar, this will clear the session and the history.

### Example Queries:

- **Chatbot:** "What are the eligibility criteria for the cost of living special payment?"
- **Chatbot:** "How much will I get if I am eligible for the special payment?"
- **Feedback:** "How many users gave a rating of 4 or higher?"
- **Feedback:** "List and explain the positive feedback that was given"


## Contributors

- Koh Jun Sheng
- Matthew Lim Wei Li
- Ben Tan Kiat
- Keith Ng Jun Hao
- Ng Jun Heng, Keith

## Additional Notes

- **Limitations:**
  - The chatbot's responses are limited by the information present in the provided documents. It cannot answer questions outside of that context.
  - The feedback analysis tool requires the `policy_feedback.txt` file to contain valid feedback in the right format.
- **Future Improvements:**
  - Add support for multiple document types beyond PDFs.
  - Implement user authentication.
  - Improve the chatbot's response formatting and contextual understanding using advanced Gemini techniques.
