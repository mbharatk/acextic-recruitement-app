# **AI-Powered Recruitment Portal**

This project enables users to streamline resume management, perform job description analysis, and find the best candidate matches using AI-driven technology. Below is a detailed guide for **setting up the tools**, even if you’re new to these technologies.

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

## **Troubleshooting**

| **Issue**                     | **Solution**                                                            |
|-------------------------------|-------------------------------------------------------------------------|
| Pinecone index not found      | Verify the index name and region in the `.env` file.                    |
| Google Drive upload fails     | Ensure the folder is shared with the service account.                   |
| OpenAI API quota exceeded     | Check your OpenAI usage or upgrade your plan.                           |
| Service account JSON missing  | Ensure the file is in the project root and named `service_account.json.` |



