"""
Cliente Supabase.
Lê SUPABASE_URL e SUPABASE_KEY de st.secrets (Streamlit Cloud),
variáveis de ambiente ou arquivo .env — nessa ordem de prioridade.
"""
import os
from threading import Lock
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None
_lock = Lock()


def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


def get_client() -> Client:
    """Retorna o cliente Supabase, criando-o na primeira chamada (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                url = _get_secret("SUPABASE_URL")
                key = _get_secret("SUPABASE_KEY")
                if not url or not key:
                    raise EnvironmentError(
                        "SUPABASE_URL e SUPABASE_KEY precisam estar definidos "
                        "em st.secrets, no .env ou nas variáveis de ambiente."
                    )
                _client = create_client(url, key)
    return _client
