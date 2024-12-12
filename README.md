# **AI-Powered Recruitment Portal**

This project enables users to streamline resume management, perform job description analysis, and find the best candidate matches using AI-driven technology. Below is a detailed guide for **setting up the tools.**

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Tools and Technologies Used](#tools-and-technologies-used)
4. [How to Get Started](#how-to-get-started)
5. [Setup Instructions](#setup-instructions)
6. [Usage](#usage)
7. [Troubleshooting](#troubleshooting)

---

## **Overview**

The **AI-Powered Recruitment Portal** offers:
- Real-time management of resumes stored in a Pinecone vector database.
- Automated resume-job description matching using OpenAI and Hugging Face models.
- Integration with Google Drive for secure file storage.
- Interactive dashboard for viewing stats and querying resumes.

This tool is ideal for recruiters and employers looking for efficient and data-driven hiring processes.

---

## **Features**

- **Resume Submission**: Candidates can submit their resumes directly through the portal.
- **Pinecone Database Integration**: Resumes are stored as vector embeddings in Pinecone for easy and efficient querying.
- **AI-Driven Matching**: Uses OpenAI embeddings and Hugging Face models to analyze and match resumes with job descriptions.
- **Dashboard**: Displays real-time stats on resumes, embedding dimensions, and growth trends.
- **Google Drive Integration**: Resumes are securely stored in a linked Google Drive folder.

---

## **Tools and Technologies Used**

| Tool/Service         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Google Drive API**  | Securely store resumes and manage uploads.                            |
| **Pinecone**          | Store and query resume embeddings in a vector database.               |
| **OpenAI API**        | Generate text embeddings for resumes and job descriptions.            |
| **Hugging Face**      | Use pre-trained models for additional embeddings.                     |
| **Google Gemini**     | Analyze resumes against job descriptions using generative AI.         |
| **Streamlit**         | Build a user-friendly web interface for the portal.                   |

---

## **How to Get Started**

Here’s how to set up and use each tool from scratch:

### **1. Google Drive API**
1. **Enable Google Drive API**:
   - Visit the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (or select an existing one).
   - Go to **APIs & Services > Library** and enable the **Google Drive API**.

2. **Create a Service Account**:
   - Go to **APIs & Services > Credentials**.
   - Click **Create Credentials > Service Account**.
   - Assign a role like **Editor** or **Owner** to the service account.

3. **Download Service Account JSON File**:
   - Once created, go to the service account and download the JSON credentials file.
   - Rename it to `service_account.json` and place it in your project folder.

4. **Share Google Drive Folder**:
   - Create a folder in Google Drive.
   - Share it with the service account email (visible in the JSON file).

---

### **2. Pinecone**
1. **Create a Pinecone Account**:
   - Sign up at [Pinecone.io](https://www.pinecone.io/).
   - After logging in, create a new index.

2. **Set Up the Index**:
   - Choose the index name (e.g., `capstone-indexed-chunks-2`).
   - Set the dimensions to match your embedding model (e.g., `1536` for OpenAI's `text-embedding-ada-002`).
   - Set the metric type to **cosine similarity**.

3. **Copy API Key and Region**:
   - Go to **API Keys** in the dashboard and copy your key.
   - Note the region (e.g., `us-east-1`) for later use.

---

### **3. OpenAI API**
1. **Create an OpenAI Account**:
   - Sign up at [OpenAI](https://platform.openai.com/).
   - Go to the **API Keys** section and generate a new key.

2. **Set Up Embedding Model**:
   - Use the embedding model `text-embedding-ada-002` in your project.

3. **Set API Key in Environment**:
   - Add your OpenAI API key to the `.env` file as `OPENAI_API_KEY`.

---

### **4. Hugging Face**
1. **Create a Hugging Face Account**:
   - Sign up at [Hugging Face](https://huggingface.co/).

2. **Install Hugging Face Transformers**:
   - Run the following command:
     ```bash
     pip install transformers
     ```

3. **Use Pre-Trained Model**:
   - The project uses `avsolatorio/NoInstruct-small-Embedding-v0`.
   - Download and use the model directly via the `transformers` library.

---

### **5. Streamlit**
1. **Install Streamlit**:
   - Run the following command:
     ```bash
     pip install streamlit
     ```

2. **Run the Application**:
   - Navigate to the project folder and run:
     ```bash
     streamlit run app.py
     ```

3. **Access the Portal**:
   - Open the provided URL (usually `http://localhost:8501`) in your browser.

---

### **6. Google Gemini Model**
1. **Get Access to Google Gemini API**:
   - Visit [Google Cloud Generative AI Studio](https://cloud.google.com/ai/generative-ai).
   - Ensure that the **Generative AI** API is enabled for your Google Cloud project.
   - Obtain an API key with access to the Gemini model.

2. **Install Required Libraries**:
   - Install the `google.generativeai` Python library:
     ```bash
     pip install google-generativeai
     ```

3. **Configure API Key**:
   - Add your Google Gemini API key to the `.env` file:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key
     ```

4. **Integrate Gemini Model into the Code**:
   - The project uses the Gemini model (`gemini-1.5-flash`) for analyzing resumes against job descriptions.
   - The integration is handled using the `google.generativeai` Python library.

5. **How Gemini Works in This Project**:
   - **Input**: 
     - A candidate’s resume (sections aggregated from Pinecone).
     - A job description (uploaded or entered by the user).
   - **Output**: 
     - Matching percentage between the resume and the job description.
     - Lists of matching and missing keywords (grouped into technical and soft skills).
     - Recommendations for improving the resume.
     - Overall assessment of the candidate’s fit for the job.

6. **Example API Usage**:
   - The project uses the following structure to interact with the Gemini model:
     ```python
     import google.generativeai as genai

     genai.configure(api_key="your_google_api_key")

     # Analyze resume and job description
     def analyze_resume_with_gemini(resume_text, job_description):
         prompt = f"""
         **Resume:**
         {resume_text}

         **Job Description:**
         {job_description}

         Provide:
         1. Matching percentage.
         2. Matching keywords grouped by technical and soft skills.
         3. Missing keywords are highlighted in bold.
         4. Recommendations for improvement.
         5. Overall assessment.
         """
         model = genai.GenerativeModel("gemini-1.5-flash")
         response = model.generate_content(prompt)
         return response.text
     ```

   - The response from the model is displayed in the **Matches** section of the portal.

---

## **Setup Instructions**

1. Clone the repository
   ```bash
   git clone https://github.com/your-repo-name/ai-recruitment-portal.git
   cd ai-recruitment-portal
   
2. Install dependencies
```bash
pip install -r requirements.txt
```
### **Configure environment variables in .env**
Create a .env file and add the following:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=us-east-1
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASSWORD=your_email_password
GOOGLE_API_KEY=your_google_api_key
GOOGLE_DRIVE_PARENT_FOLDER_ID=your_google_drive_folder_id
```
### **Start the application**
```bash
streamlit run app.py
```

## **Usage**

### **Submit Resume**
- Candidates upload resumes before logging in.

### **Log In**
- Employers and recruiters log in to access advanced features.

### **Dashboard**
- View real-time stats (e.g., total resumes indexed).

### **Search & Filter**
- Query resumes using skills or job descriptions.

### **Matches**
- Find and analyze the best candidates for a job description.


---

# Backend Implementation for AI-Driven Recruitment Tool

## Tools and Frameworks Used

### Programming Languages
- **Python**: Version 3.8 or higher.

### Libraries
- **openai**: For GPT-based resume parsing and embedding generation.
- **transformers**: For embedding generation using Hugging Face models.
- **pinecone-client**: For storing and querying vector embeddings.
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For evaluation metrics like precision and recall and nearest neighbor search.
- **tiktoken**: For token processing with OpenAI APIs.
- **chromadb**: For managing embeddings in a lightweight database.

### Models
- **OpenAI GPT-3.5 Turbo**: For resume parsing.
- **Hugging Face Model**: `NoInstruct-small-Embedding-v0` for embedding generation.

### Databases
- **Google Drive**: For storing raw resumes and extracted JSON files.
- **Pinecone**: For indexing and querying vector embeddings.
- **ChromaDB**: For managing embeddings and metadata.

### Environment
- **Google Colab**: For running the backend scripts and accessing Google Drive.

### Visualization
- **matplotlib**: For generating precision-recall curves and other plots.

---

## Features

### **1. Resume Collection and Storage**
1. **File Upload**:
   - Users can upload resumes via an intuitive interface supporting drag-and-drop or file picker functionality.
   - Real-time feedback is provided for successful uploads and file processing.

2. **Resume Parsing**:
   - Uploaded resumes are automatically parsed using OpenAI GPT-3.5 Turbo.
   - Key information such as name, education, employment history, and skills are extracted.

3. **Storage**:
   - Parsed resumes are saved in Google Drive in JSON format for further processing.
   - Raw resumes are stored in a separate directory for reference.

---

### **2. Backend Processing and Embedding**

1. **Resume Embedding**:
   - Parsed resumes are divided into meaningful chunks:
     - `education`
     - `employment`
     - `technical_skills`
     - `soft_skills`
   - Each chunk is converted into a vector embedding using Hugging Face's `NoInstruct-small-Embedding-v0` model.

2. **Metadata Assignment**:
   - Metadata such as tags (`IT`, `Non-IT`, keywords) and filenames are generated for each chunk to improve search precision.

3. **Storage in Pinecone**:
   - Embeddings are stored in Pinecone, a cloud-based vector database.
   - The index is configured for cosine similarity and deployed in the `us-east-1` region.
   - Combined embeddings (all sections) and section-specific embeddings are stored for efficient matching.

4. **Job Description Matching**:
   - A job description provided by the user is embedded using the same Hugging Face model.
   - The embedded query vector is compared with stored resume vectors in Pinecone using cosine similarity.
   - Top matches are retrieved with their similarity scores and metadata.

---

### **3. Evaluation and Visualization**

1. **Precision-Recall Analysis**:
   - Evaluation metrics such as precision, recall, and Mean Reciprocal Rank (MRR) are computed for matching results.
   - Precision-recall curves and threshold-based precision curves are visualized to assess system performance.

2. **IT vs Non-IT Classification**:
   - Resumes are classified as IT or Non-IT based on thresholds applied to similarity scores and metadata.

3. **Interactive Adjustments**:
   - Users can interactively adjust thresholds in the UI to refine the classification and matching accuracy.

---

## Setup Instructions

### Prerequisites
1. Install Python (version 3.8 or higher).
2. Install required libraries:
   ```bash
   pip install openai transformers pandas scikit-learn matplotlib pinecone-client chromadb tiktoken
   ```

### Step 1: Prepare the Environment
1. Mount Google Drive if using Colab:
    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    ```
2. Place raw resume files in:
    ```
    /content/drive/My Drive/Resumes_text_files/
    ```

---

### Step 2: Parse Resumes
Run the script to parse resumes:
```python
process_resumes(
    directory_path="/content/drive/My Drive/Resumes_text_files/",
    output_directory="/content/drive/My Drive/Extracted_Resumes_final/"
)
```

---

## Step 3: Generate Embeddings

Generate embeddings for parsed resumes:

```python
resume_vectors = process_resumes_and_vectorize(
    folder_path="/content/drive/My Drive/Extracted_Resumes_final/"
)
```

---

## Step 4: Store Data in Pinecone

### Initialize Pinecone and Create an Index
To store embeddings, first initialize Pinecone and create an index:

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with your API key
pc = Pinecone(api_key="your_pinecone_api_key")

# Create an index with the specified configuration
pc.create_index(
    name="capstone-indexed-chunks",
    dimension=1536,  # Dimensionality of the embedding vectors
    metric="cosine",  # Metric used for similarity search
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Deployment configuration
)
```

### Store Embeddings in the Index

Once the index is ready, store the resume embeddings along with metadata:

```python
index = pc.Index("capstone-indexed-chunks")

# Loop through the DataFrame to upload embeddings
for idx, row in final_df.iterrows():
    filename = row['filename']
    combined_vector = np.concatenate([
        np.mean(row['education'], axis=0),
        np.mean(row['employment'], axis=0),
        np.mean(row['technical_skills'], axis=0),
        np.mean(row['soft_skills'], axis=0)
    ]).tolist()

    tags = row['tags'].split(",")  # Extract tags as metadata

    # Upsert the vector and metadata into Pinecone
    index.upsert([
        (filename, combined_vector, {
            "filename": filename,
            "tags": tags
        })
    ])

print("Embeddings successfully uploaded to Pinecone!")
```

#### Summary

- **Purpose**: Pinecone is used to store high-dimensional vectors and perform similarity searches.
- **Metadata**: Each embedding is stored with associated metadata (e.g., filename and tags).
- **Index Configuration**:
  - **Metric**: Cosine similarity.
  - **Region**: `us-east-1`.
- **Note**: Replace `"your_pinecone_api_key"` with your actual Pinecone API key.

---

## Step 5: Query and Match Resumes

### Input a Job Description
Provide a job description to generate a query embedding:

```python
job_description = "Looking for a candidate with a Master's degree in Computer Science"
query_vector = get_embedding(job_description)
```

### Query the Database
Use the query embedding to search the Pinecone index for the top matching resumes:

```python
results = index.query(
    vector=query_vector,
    top_k=5,  # Number of top matches to retrieve
    include_metadata=True  # Include metadata like filename and tags in the results
)
```

---

## Step 6: Evaluate Performance

### Run evaluation metrics:

```python
precision = precision_score(df_evl['Actual'], df_evl['Predicted'])
recall = recall_score(df_evl['Actual'], df_evl['Predicted'])
```

### Generate a precision-recall curve:

```python
plt.plot(recalls, precisions, marker='o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

## Future Enhancements

- **Integrate Real-Time Data Ingestion**: Enable automated ingestion of new resumes and job descriptions to keep the database up-to-date.
- **Explore Advanced Contextual Embeddings**: Implement more sophisticated embedding models for improved semantic alignment between resumes and job descriptions.
- **Develop a User-Friendly Interface**: Create an intuitive interface for recruiters to query the system and visualize matching results effectively.

---

## **Troubleshooting**

| **Issue**                     | **Solution**                                                            |
|-------------------------------|-------------------------------------------------------------------------|
| Pinecone index not found      | Verify the index name and region in the `.env` file.                    |
| Google Drive upload fails     | Ensure the folder is shared with the service account.                   |
| OpenAI API quota exceeded     | Check your OpenAI usage or upgrade your plan.                           |
| Service account JSON missing  | Ensure the file is in the project root and named `service_account.json.` |


**if there are any issues with using the libraries or frameworks, make sure all the following imports are available in their respective codes, if not please add them**
#### Backend Imports
- The backend handles tasks like resume parsing, embedding generation, and Pinecone database integration.

```python
# General Utilities
import os
import json
import numpy as np
import pandas as pd

# OpenAI API for resume parsing and embeddings
import openai

# Hugging Face Transformers for embedding generation
from transformers import AutoModel, AutoTokenizer
import torch

# Pinecone for vector storage and querying
from pinecone import Pinecone, ServerlessSpec

# Evaluation Metrics and Visualization
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Tokenizer for OpenAI embeddings
import tiktoken

# ChromaDB for lightweight vector database management
import chromadb

# Scikit-learn for Nearest Neighbors
from sklearn.neighbors import NearestNeighbors

# Cosine similarity for vector comparisons
from sklearn.metrics.pairwise import cosine_similarity
```

#### UI Imports
- The UI focuses on building an interactive Streamlit app for uploading resumes, job description input, and result visualization.

```python
# Streamlit for building the interactive UI
import streamlit as st

# Data manipulation and processing
import pandas as pd
import numpy as np

# Environment variable management
from dotenv import load_dotenv

# Pinecone for vector storage and querying
from pinecone import Pinecone, ServerlessSpec

# Google Generative AI (e.g., Bard) integration
import google.generativeai as genai

# File and cloud interaction
import os
import json
import tempfile

# Hugging Face for embedding generation
import torch
from transformers import AutoModel, AutoTokenizer

# Google APIs for Drive and Gmail
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload

# Email functionality
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Visualization for displaying results
import plotly.express as px
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

