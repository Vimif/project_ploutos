"""
LLM Explainer : genere des explications en langage naturel via Ollama.

Fallback sur des templates si Ollama n'est pas disponible.
Suit le pattern de degradation gracieuse du projet (OLLAMA_AVAILABLE).
"""

import logging
from typing import Dict

from advisory.llm.prompts import (
    SYSTEM_PROMPT_FR,
    format_analysis_prompt,
    generate_fallback_explanation,
)

logger = logging.getLogger(__name__)

# Import optionnel Ollama
try:
    import ollama as ollama_lib

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.info("ollama non installe, explications LLM desactivees")


class LLMExplainer:
    """Genere des explications en francais via Ollama ou fallback template."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = False

        if OLLAMA_AVAILABLE:
            self._available = self._check_ollama()

        if self._available:
            logger.info(f"LLMExplainer: Ollama disponible (modele: {self.model})")
        else:
            logger.info("LLMExplainer: Fallback sur templates (Ollama indisponible)")

    def _check_ollama(self) -> bool:
        """Verifie si Ollama est en cours d'execution et le modele disponible."""
        try:
            client = ollama_lib.Client(host=self.base_url)
            models = client.list()
            model_names = [m.model for m in models.models] if models.models else []
            if any(self.model in name for name in model_names):
                return True
            logger.warning(
                f"Modele '{self.model}' non trouve dans Ollama. "
                f"Disponibles: {model_names}"
            )
            return False
        except Exception as e:
            logger.warning(f"Ollama non accessible: {e}")
            return False

    def explain(self, analysis_data: Dict) -> str:
        """
        Genere une explication en francais pour un resultat d'analyse.

        Args:
            analysis_data: Dictionnaire de l'AdvisoryResult (via to_dict())

        Returns:
            Explication en francais (LLM ou fallback template)
        """
        if self._available:
            try:
                return self._generate_llm(analysis_data)
            except Exception as e:
                logger.error(f"Erreur LLM, fallback sur template: {e}")

        return generate_fallback_explanation(analysis_data)

    def _generate_llm(self, analysis_data: Dict) -> str:
        """Appelle Ollama pour generer l'explication."""
        prompt = format_analysis_prompt(analysis_data)

        client = ollama_lib.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FR},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.message.content
        if content:
            return content.strip()

        logger.warning("Reponse LLM vide, fallback sur template")
        return generate_fallback_explanation(analysis_data)

    @property
    def is_available(self) -> bool:
        return self._available
