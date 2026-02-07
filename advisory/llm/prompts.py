"""
Templates de prompts pour le LLM (Ollama/Mistral).

Tous les prompts sont en francais pour generer des explications
en langage naturel destinees a l'utilisateur.
"""

SYSTEM_PROMPT_FR = """Tu es un conseiller financier expert. Tu analyses des actions boursieres
et fournis des explications claires en francais. Tu ne donnes PAS de conseil d'investissement
personnalise, tu expliques les indicateurs techniques, le sentiment de marche, et les
previsions statistiques de maniere pedagogique.

Regles:
- Commence par le score global et la recommandation
- Explique chaque source d'analyse (technique, ML, sentiment, prevision) en 2-3 phrases
- Termine par les niveaux cles (support, resistance, stop-loss, take-profit)
- Sois factuel et base tes explications uniquement sur les donnees fournies
- Utilise un ton professionnel mais accessible
- Reponds en 200-300 mots maximum
- Langue: francais uniquement"""


ANALYSIS_TEMPLATE = """Analyse pour {symbol} au {date}:

Score composite: {composite_score:.2f}/1.00 -> {recommendation}
Prix actuel: ${current_price:.2f}

ANALYSE TECHNIQUE (signal: {tech_signal:+.2f}, confiance: {tech_confidence:.0%}):
{tech_reasons}

PREDICTION IA (signal: {ml_signal:+.2f}, confiance: {ml_confidence:.0%}):
{ml_reasons}

SENTIMENT MARCHE (signal: {sent_signal:+.2f}, confiance: {sent_confidence:.0%}):
{sent_reasons}

PREVISION STATISTIQUE (signal: {stat_signal:+.2f}, confiance: {stat_confidence:.0%}):
{stat_reasons}

RISQUE (signal: {risk_signal:+.2f}):
{risk_reasons}

{price_levels}

Genere une explication complete en francais pour un investisseur particulier."""


def format_analysis_prompt(analysis_data: dict) -> str:
    """Formate le template avec les donnees d'analyse."""
    # Extraire les sous-signaux par source
    sub_signals = {s["source"]: s for s in analysis_data.get("sub_signals", [])}

    tech = sub_signals.get("technical", {})
    ml = sub_signals.get("ml_model", {})
    sent = sub_signals.get("sentiment", {})
    stat = sub_signals.get("statistical", {})
    risk = sub_signals.get("risk", {})

    # Niveaux de prix
    price_levels = ""
    tech_details = tech.get("details", {})
    if tech_details.get("entry_price"):
        price_levels += f"Prix d'entree: ${tech_details['entry_price']:.2f}\n"
    if tech_details.get("stop_loss"):
        price_levels += f"Stop-loss: ${tech_details['stop_loss']:.2f}\n"
    if tech_details.get("take_profit"):
        price_levels += f"Take-profit: ${tech_details['take_profit']:.2f}\n"

    return ANALYSIS_TEMPLATE.format(
        symbol=analysis_data.get("symbol", "???"),
        date=analysis_data.get("timestamp", "")[:10],
        composite_score=analysis_data.get("composite_score", 0),
        recommendation=analysis_data.get("recommendation", "NEUTRE"),
        current_price=analysis_data.get("current_price", 0),
        tech_signal=tech.get("signal", 0),
        tech_confidence=tech.get("confidence", 0),
        tech_reasons="\n".join(f"- {r}" for r in tech.get("reasons", ["N/A"])),
        ml_signal=ml.get("signal", 0),
        ml_confidence=ml.get("confidence", 0),
        ml_reasons="\n".join(f"- {r}" for r in ml.get("reasons", ["N/A"])),
        sent_signal=sent.get("signal", 0),
        sent_confidence=sent.get("confidence", 0),
        sent_reasons="\n".join(f"- {r}" for r in sent.get("reasons", ["N/A"])),
        stat_signal=stat.get("signal", 0),
        stat_confidence=stat.get("confidence", 0),
        stat_reasons="\n".join(f"- {r}" for r in stat.get("reasons", ["N/A"])),
        risk_signal=risk.get("signal", 0),
        risk_reasons="\n".join(f"- {r}" for r in risk.get("reasons", ["N/A"])),
        price_levels=price_levels or "Niveaux de prix non disponibles.",
    )


# --- Fallback template (sans LLM) ---

RECOMMENDATION_LABELS = {
    "ACHAT_FORT": "Achat fort",
    "ACHAT": "Achat",
    "NEUTRE": "Neutre / Attendre",
    "VENTE": "Vente",
    "VENTE_FORTE": "Vente forte",
}


def generate_fallback_explanation(analysis_data: dict) -> str:
    """Genere une explication template (sans LLM) en francais."""
    symbol = analysis_data.get("symbol", "???")
    score = analysis_data.get("composite_score", 0)
    rec = analysis_data.get("recommendation", "NEUTRE")
    rec_label = RECOMMENDATION_LABELS.get(rec, rec)
    price = analysis_data.get("current_price", 0)

    sub_signals = {s["source"]: s for s in analysis_data.get("sub_signals", [])}

    parts = [
        f"**{symbol}** - Recommandation : **{rec_label}** (score {score:+.2f})",
        f"Prix actuel : ${price:.2f}",
        "",
    ]

    # Section par source
    source_labels = {
        "technical": "Analyse technique",
        "ml_model": "Intelligence artificielle",
        "sentiment": "Sentiment du marche",
        "statistical": "Prevision statistique",
        "risk": "Evaluation du risque",
    }

    for source_key, label in source_labels.items():
        sub = sub_signals.get(source_key)
        if sub and sub.get("confidence", 0) > 0:
            sig = sub["signal"]
            direction = "haussier" if sig > 0.1 else "baissier" if sig < -0.1 else "neutre"
            parts.append(
                f"**{label}** ({sub['confidence']:.0%} confiance) : Signal {direction} ({sig:+.2f})"
            )
            for reason in sub.get("reasons", [])[:2]:
                parts.append(f"  - {reason}")
            parts.append("")

    # Niveaux de prix
    tech = sub_signals.get("technical", {})
    tech_details = tech.get("details", {})
    if tech_details.get("stop_loss"):
        parts.append(
            f"Stop-loss suggere : ${tech_details['stop_loss']:.2f} | "
            f"Take-profit : ${tech_details.get('take_profit', 0):.2f}"
        )

    # Prevision
    forecast = analysis_data.get("forecast", [])
    if forecast:
        last = forecast[-1]
        if isinstance(last, dict):
            parts.append(
                f"Prevision J+5 : ${last.get('predicted', 0):.2f} "
                f"(intervalle 80% : ${last.get('lower_80', 0):.2f} - "
                f"${last.get('upper_80', 0):.2f})"
            )

    return "\n".join(parts)
