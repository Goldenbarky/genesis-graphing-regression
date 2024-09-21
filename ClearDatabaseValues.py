from supabase import create_client, Client

url: str = "https://kdubupeizubmgkwgcguu.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtkdWJ1cGVpenVibWdrd2djZ3V1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNjMyMzg4MCwiZXhwIjoyMDQxODk5ODgwfQ._PMoTOlc2D53dn7Mmr0j5Ka_UoruxVK4TilzpflXd08"
supabase: Client = create_client(url, key)

supabase.table("eod_equations").delete().neq('eq_data', 0).execute()
supabase.table("data").delete().neq('value', -1).execute()