"""
LLM module — Ollama integration for feature extraction and entity resolution.

All functions are non-fatal: if Ollama is unavailable the pipeline continues
with zero-filled LLM features. The XGBoost v3 model ignores these features
(it only reads its 27 feature names); they are stored for v4 training.

Environment variables:
  OLLAMA_HOST        default: http://localhost:11434
  OLLAMA_TEXT_MODEL  default: llama3.2   (used for text analysis)
  OLLAMA_EMBED_MODEL default: nomic-embed-text (used for embeddings)
"""
from llm.client import OllamaClient, get_client
from llm.text_features import extract_llm_features, LLMFeatures
from llm.entity_resolver import resolve_entity, EntityCanon

__all__ = [
    "OllamaClient",
    "get_client",
    "extract_llm_features",
    "LLMFeatures",
    "resolve_entity",
    "EntityCanon",
]
