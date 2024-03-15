from dotenv import load_dotenv
import os 

#loads env vars located in .env file
load_dotenv()

#get client id and secret
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

print(client_id, client_secret)