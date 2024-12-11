AI-Powered Recruitment Portal
This project enables users to streamline resume management, perform job description analysis, and find the best candidate matches using AI-driven technology. Below is a detailed guide for setting up the tools, even if you’re new to these technologies.

Table of Contents
Overview
Features
Tools and Technologies Used
How to Get Started
Setup Instructions
Usage
Troubleshooting
Overview
The AI-Powered Recruitment Portal offers:

Real-time management of resumes stored in a Pinecone vector database.
Automated resume-job description matching using OpenAI and Hugging Face models.
Integration with Google Drive for secure file storage.
Interactive dashboard for viewing stats and querying resumes.
This tool is ideal for recruiters and employers looking for efficient and data-driven hiring processes.

Features
Resume Submission: Candidates can submit their resumes directly through the portal.
Pinecone Database Integration: Resumes are stored as vector embeddings in Pinecone for easy and efficient querying.
AI-Driven Matching: Uses OpenAI embeddings and Hugging Face models to analyze and match resumes with job descriptions.
Dashboard: Displays real-time stats on resumes, embedding dimensions, and growth trends.
Google Drive Integration: Resumes are securely stored in a linked Google Drive folder.
Tools and Technologies Used
Tool/Service	Purpose
Google Drive API	Securely store resumes and manage uploads.
Pinecone	Store and query resume embeddings in a vector database.
OpenAI API	Generate text embeddings for resumes and job descriptions.
Hugging Face	Use pre-trained models for additional embeddings.
Streamlit	Build a user-friendly web interface for the portal.
How to Get Started
Here’s how to set up and use each tool from scratch:

1. Google Drive API
Enable Google Drive API:

Visit the Google Cloud Console.
Create a new project (or select an existing one).
Go to APIs & Services > Library and enable the Google Drive API.
Create a Service Account:

Go to APIs & Services > Credentials.
Click Create Credentials > Service Account.
Assign a role like Editor or Owner to the service account.
Download Service Account JSON File:

Once created, go to the service account and download the JSON credentials file.
Rename it to service_account.json and place it in your project folder.
Share Google Drive Folder:

Create a folder in Google Drive.
Share it with the service account email (visible in the JSON file).
2. Pinecone
Create a Pinecone Account:

Sign up at Pinecone.io.
After logging in, create a new index.
Set Up the Index:

Choose the index name (e.g., capstone-indexed-chunks-2).
Set the dimensions to match your embedding model (e.g., 1536 for OpenAI's text-embedding-ada-002).
Set the metric type to cosine similarity.
Copy API Key and Region:

Go to API Keys in the dashboard and copy your key.
Note the region (e.g., us-east-1) for later use.
3. OpenAI API
Create an OpenAI Account:

Sign up at OpenAI.
Go to the API Keys section and generate a new key.
Set Up Embedding Model:

Use the embedding model text-embedding-ada-002 in your project.
Set API Key in Environment:

Add your OpenAI API key to the .env file as OPENAI_API_KEY.
4. Hugging Face
Create a Hugging Face Account:

Sign up at Hugging Face.
Install Hugging Face Transformers:

Run the following command:
bash
Copy code
pip install transformers
Use Pre-Trained Model:

The project uses avsolatorio/NoInstruct-small-Embedding-v0.
Download and use the model directly via the transformers library.
5. Streamlit
Install Streamlit:

Run the following command:
bash
Copy code
pip install streamlit
Run the Application:

Navigate to the project folder and run:
bash
Copy code
streamlit run app.py
Access the Portal:

Open the provided URL (usually http://localhost:8501) in your browser.
Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo-name/ai-recruitment-portal.git
cd ai-recruitment-portal
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Configure environment variables in .env:

plaintext
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=us-east-1
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASSWORD=your_email_password
GOOGLE_API_KEY=your_google_api_key
GOOGLE_DRIVE_PARENT_FOLDER_ID=your_google_drive_folder_id
Start the application:

bash
Copy code
streamlit run app.py
Usage
Submit Resume:

Candidates upload resumes before logging in.
Log In:

Employers and recruiters log in to access advanced features.
Dashboard:

View real-time stats (e.g., total resumes indexed).
Search & Filter:

Query resumes using skills or job descriptions.
Matches:

Find and analyze the best candidates for a job description.
Troubleshooting
Issue	Solution
Pinecone index was not found. Verify the index name and region in the .env file.
Google Drive upload fails to	Ensure the folder is shared with the service account.
OpenAI API quota exceeded	Check your OpenAI usage or upgrade your plan.
Service account JSON missing	Ensure the file is in the project root and named service_account.json.
This markdown format is ready to use. Copy it into your README.md file and adjust repository-specific details where needed. Let me know if you need further help!
