"""
Ingesta o arquivo data/Obesity.csv na tabela `obesity_data` do Supabase.

Uso:
    python -m db.ingest_csv

Pré-requisito:
    Executar db/create_tables.sql no SQL Editor do Supabase antes da primeira ingestão.
"""
import pandas as pd
from pathlib import Path
from db.supabase_client import get_client

CSV_PATH = Path(__file__).parent.parent / "data" / "Obesity.csv"
TABELA   = "obesity_data"
BATCH    = 500  # linhas por chamada de upsert


def ingest() -> None:
    df = pd.read_csv(CSV_PATH)
    # Normaliza nomes de coluna para minúsculas com underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    registros = df.to_dict(orient="records")
    client    = get_client()

    for i in range(0, len(registros), BATCH):
        lote = registros[i : i + BATCH]
        client.table(TABELA).upsert(lote).execute()
        print(f"Inseridas linhas {i} – {i + len(lote) - 1}")

    print(f"Concluído. {len(registros)} linhas ingeridas na tabela '{TABELA}'.")


if __name__ == "__main__":
    ingest()
