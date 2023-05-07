
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from google.oauth2 import service_account
import base64

def send_email(subject, body, to):
    # Set up Gmail API credentials
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    CLIENT_SECRET_FILE = 'secret.json'
    # SERVICE_ACCOUNT_FILE = 'service_secret.json'
    
    # Load saved credentials or obtain new ones
    creds = None
    try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    except FileNotFoundError:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        # flow = InstalledAppFlow.from_client_secrets_file(SERVICE_ACCOUNT_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Create the email message
    message = MIMEMultipart()
    message['to'] = to
    message['subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    
    try:
        # Call the Gmail API to send the email
        service = build('gmail', 'v1', credentials=creds)
        send_message = (service.users().messages().send(userId="me", body=create_message).execute())
        print(F'Email was sent to {to} with Email Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message