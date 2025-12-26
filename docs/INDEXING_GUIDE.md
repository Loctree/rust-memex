# rmcp-memex Indexing Guide

> Przewodnik dla agentów AI do indeksowania wspomnień i dokumentów.

## Quick Start

```bash
# Indeksuj pojedynczy plik
rmcp_memex index --db-path /path/to/lancedb --namespace moje-wspomnienia /path/to/file.json

# Indeksuj cały folder
rmcp_memex index --db-path /path/to/lancedb --namespace pamietnik /path/to/folder/

# Z filtrem na rozszerzenia
rmcp_memex index --db-path /path/to/lancedb --namespace rozmowy --glob "*.json" /path/to/folder/
```

## Wymagania

1. **Embedder musi działać** na `localhost:12345`
   - Sprawdź: `curl http://localhost:12345/health`
   - Oczekiwany model: `Qwen3-Embedding-8B` (4096 dims)

2. **LanceDB path** musi istnieć i być zapisywalny

## Namespace - klucz do organizacji

Namespace to logiczna izolacja danych. Wybieraj nazwy mówiące o zawartości:

| Namespace | Zawartość |
|-----------|-----------|
| `pamietnik` | Pełna historia rozmów |
| `projekty` | Dokumentacja projektów |
| `kod` | Snippety kodu |
| `wiedza` | Artykuły, notatki |

```bash
# Różne namespace'y dla różnych typów danych
rmcp_memex index --namespace rozmowy-2024 /data/conversations_2024.json
rmcp_memex index --namespace rozmowy-2025 /data/conversations_2025.json
```

## Obsługiwane formaty

| Format | Obsługa |
|--------|---------|
| `.json` | ✓ Conversation exports |
| `.md` | ✓ Markdown |
| `.txt` | ✓ Plain text |
| `.pdf` | ✓ PDF extraction |

## Chunking

rmcp-memex automatycznie dzieli tekst na chunki:
- **Rozmiar:** 512 znaków
- **Overlap:** 128 znaków (kontekst między chunkami)

## Workflow dla dużych zbiorów

```bash
# 1. Sprawdź co masz do zaindeksowania
find /path/to/data -type f \( -name "*.json" -o -name "*.md" \) | wc -l

# 2. Zacznij od małej próbki
rmcp_memex index --namespace test --glob "*.json" /path/to/data/sample/

# 3. Sprawdź wyniki przez MCP
# (użyj rag_search z namespace="test")

# 4. Jeśli OK - indeksuj wszystko
rmcp_memex index --namespace produkcja /path/to/data/
```

## Szukanie po zaindeksowaniu

Przez MCP (dla agentów):
```
rag_search(query="szukana fraza", namespace="pamietnik", k=5)
```

## Troubleshooting

### "Access denied: path outside allowed directories"
CLI nie ma tego ograniczenia. Jeśli widzisz ten błąd, używasz MCP zamiast CLI.

### Indeksowanie trwa długo
To normalne dla dużych plików. 8MB JSON = ~15-20 minut z Qwen3-Embedding-8B.

### Puste wyniki wyszukiwania
1. Sprawdź czy podajesz właściwy namespace
2. Sprawdź czy embedder działa
3. Sprawdź wymiary wektorów (muszą się zgadzać: 4096)

---

Created by M&K (c)2025 The LibraxisAI Team
