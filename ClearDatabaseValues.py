from supabase import create_client, Client
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

supabase.table("eod_equations").delete().neq('eq_data', 0).execute()
supabase.table("data").delete().neq('value', -1).execute()