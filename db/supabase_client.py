import os
from threading import Lock
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None
_lock = Lock()


def _read_secrets() -> tuple[str, str]:
    """Lê credenciais de st.secrets (Cloud) ou variáveis de ambiente (.env/sistema)."""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if url and key:
            return url, key
    except Exception:
        pass
    return os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_KEY", "")


def get_client() -> Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                url, key = _read_secrets()
                if not url or not key:
                    raise EnvironmentError(
                        "SUPABASE_URL e SUPABASE_KEY precisam estar definidos "
                        "em st.secrets, no .env ou nas variáveis de ambiente."
                    )
                _client = create_client(url, key)
    return _client
