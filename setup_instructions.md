# Gmail Agent Setup Instructions

## Prerequisites
1. Python 3.7+
2. OpenAI API key (for LangChain agent)

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Google Cloud Console
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API" and enable it
4. Create OAuth2 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application"
   - Download the JSON file and save as `credentials.json` in this directory

### 3. Environment Setup
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to the `.env` file

### 4. Run the Agent
```bash
python gmail_agent.py
```

## First Run
- The first time you run the script, it will open a browser for OAuth authentication
- Grant permission to access your Gmail account
- The token will be saved for future use

## Security Notes
- Your Gmail credentials are stored locally and never shared
- The script only requests permission to modify Gmail messages (mark as read)
- All authentication happens through Google's official OAuth2 flow