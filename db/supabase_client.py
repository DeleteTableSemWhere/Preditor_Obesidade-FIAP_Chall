"""
Cliente Supabase,
Lê SUPABASE_URL e SUPABASE_KEY do arquivo .env ou variáveis de ambiente.
"""
import os
from threading import Lock
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None
_lock = Lock()


def get_client() -> Client:
    """Retorna o cliente Supabase, criando-o na primeira chamada (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                url = os.environ.get("SUPABASE_URL", "")
                key = os.environ.get("SUPABASE_KEY", "")
                if not url or not key:
                    raise EnvironmentError(
                        "SUPABASE_URL e SUPABASE_KEY precisam estar definidos no .env ou no ambiente."
                    )
                _client = create_client(url, key)
    return _client
