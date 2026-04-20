-- Migrações aplicadas após a criação inicial das tabelas (create_tables.sql)
-- Execute no SQL Editor do Supabase

-- Índice para ordenação cronológica no histórico de predições
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions_log (created_at DESC);

-- Índice para filtros e agrupamentos por classe predita
CREATE INDEX IF NOT EXISTS idx_predictions_class
    ON predictions_log (predicted_class);
