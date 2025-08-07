import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

class GmailService:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API using OAuth2"""
        creds = None
        
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Please download your OAuth2 credentials from Google Cloud Console and save as '{self.credentials_file}'")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Successfully authenticated with Gmail API")
    
    def get_old_emails(self, days_old: int = 365) -> List[Dict[str, Any]]:
        """Get emails older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        query = f'before:{cutoff_date.strftime("%Y/%m/%d")} is:unread'
        
        try:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=500).execute()
            messages = results.get('messages', [])
            
            logger.info(f"Found {len(messages)} unread emails older than {days_old} days")
            return messages
        
        except Exception as e:
            logger.error(f"Error fetching old emails: {e}")
            return []
    
    def mark_emails_as_read(self, message_ids: List[str]) -> bool:
        """Mark multiple emails as read"""
        if not message_ids:
            return True
        
        try:
            batch_size = 100
            for i in range(0, len(message_ids), batch_size):
                batch = message_ids[i:i + batch_size]
                
                self.service.users().messages().batchModify(
                    userId='me',
                    body={
                        'ids': batch,
                        'removeLabelIds': ['UNREAD']
                    }
                ).execute()
                
                logger.info(f"Marked {len(batch)} emails as read")
            
            logger.info(f"Successfully marked {len(message_ids)} emails as read")
            return True
        
        except Exception as e:
            logger.error(f"Error marking emails as read: {e}")
            return False

class GmailAgent:
    def __init__(self):
        self.gmail_service = GmailService()
        self.agent = self._create_agent()
    
    def _get_old_emails_tool(self, days_old: str = "365") -> str:
        """Tool to get old emails"""
        try:
            days = int(days_old)
            emails = self.gmail_service.get_old_emails(days)
            return f"Found {len(emails)} unread emails older than {days} days"
        except Exception as e:
            return f"Error: {e}"
    
    def _mark_emails_read_tool(self, days_old: str = "365") -> str:
        """Tool to mark old emails as read"""
        try:
            days = int(days_old)
            emails = self.gmail_service.get_old_emails(days)
            
            if not emails:
                return "No old unread emails found"
            
            message_ids = [msg['id'] for msg in emails]
            success = self.gmail_service.mark_emails_as_read(message_ids)
            
            if success:
                return f"Successfully marked {len(message_ids)} emails as read"
            else:
                return "Failed to mark emails as read"
        
        except Exception as e:
            return f"Error: {e}"
    
    def _create_agent(self):
        """Create LangChain agent with Gmail tools"""
        tools = [
            Tool(
                name="get_old_emails",
                description="Get count of unread emails older than specified days (default 365)",
                func=self._get_old_emails_tool
            ),
            Tool(
                name="mark_emails_read",
                description="Mark all unread emails older than specified days as read (default 365)",
                func=self._mark_emails_read_tool
            )
        ]
        
        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""You are a Gmail management assistant. You can help mark old emails as read.

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: think about what you need to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
        )
        
        llm = OpenAI(temperature=0)
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def process_old_emails(self, days_old: int = 365):
        """Process old emails using the agent"""
        query = f"Mark all unread emails older than {days_old} days as read"
        result = self.agent.invoke({"input": query})
        return result

def main():
    """Main function to run the Gmail agent"""
    try:
        agent = GmailAgent()
        result = agent.process_old_emails(365)
        print(f"Result: {result['output']}")
    
    except FileNotFoundError as e:
        print(f"Setup required: {e}")
        print("\nTo get started:")
        print("1. Go to Google Cloud Console")
        print("2. Enable Gmail API")
        print("3. Create OAuth2 credentials")
        print("4. Download and save as 'credentials.json'")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()