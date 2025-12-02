# llm_analyst.py
# ---------------------------------------------------------
# DIRECTEUR FINANCIER LOCAL (OLLAMA / LLAMA 3.2)
# ---------------------------------------------------------
import ollama
import yfinance as yf
import re

# Assurez-vous d'avoir installé Ollama (https://ollama.com)
# Et d'avoir lancé 'ollama run llama3.2' dans un terminal au moins une fois.
MODEL_NAME = "llama3.2" 

def ask_the_oracle(ticker):
    """
    Analyse les news récentes avec Llama 3.2 localement.
    Retourne un tuple : (Verdict, Explication)
    
    Verdict possibles : "BULLISH", "BEARISH", "NEUTRAL"
    """
    print(f"🧠 [Llama] Analyse sémantique des news sur {ticker}...")
    
    # 1. Récupération des News (Yahoo Finance)
    try:
        t = yf.Ticker(ticker)
        news = t.news
        
        if not news:
            # Pas de news = Pas d'avis (Neutre)
            return "NEUTRAL", "Pas de news récentes disponibles."
        
        # On récupère les 5 titres les plus récents + un bout du résumé si dispo
        headlines = []
        for n in news[:5]:
            titre = n.get('title', '')
            headlines.append(f"- {titre}")
            
        text_block = "\n".join(headlines)
        
    except Exception as e:
        return "NEUTRAL", f"Erreur Yahoo Finance: {e}"

    # 2. Le Prompt (L'ordre donné au cerveau)
    # On demande une réponse très structurée pour pouvoir la parser facilement
    prompt = f"""
    Agis comme un analyste financier senior.
    Voici les derniers titres concernant l'action {ticker} :
    
    {text_block}
    
    Tâche : Analyse le sentiment de ces titres. Sont-ils positifs ou négatifs pour le prix de l'action à court terme ?
    
    Réponds uniquement avec ce format exact :
    VERDICT: [BULLISH ou BEARISH ou NEUTRAL]
    RAISON: [Une phrase courte d'explication en français]
    """

    # 3. Appel à Ollama (Local)
    try:
        # Appel à l'API locale d'Ollama (port 11434 par défaut)
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        content = response['message']['content']
        
        # 4. Analyse de la réponse (Parsing)
        verdict = "NEUTRAL"
        reason = content
        
        content_upper = content.upper()
        
        if "VERDICT: BULLISH" in content_upper:
            verdict = "BULLISH"
        elif "VERDICT: BEARISH" in content_upper:
            verdict = "BEARISH"
        elif "VERDICT: NEUTRAL" in content_upper:
            verdict = "NEUTRAL"
        else:
            # Si l'IA n'a pas respecté le format, on cherche les mots clés
            if "BULLISH" in content_upper: verdict = "BULLISH"
            elif "BEARISH" in content_upper: verdict = "BEARISH"
        
        # Extraction de la raison propre
        if "RAISON:" in content:
            parts = content.split("RAISON:")
            if len(parts) > 1:
                reason = parts[1].strip().split('\n')[0] # On prend juste la ligne
        else:
            reason = content[:100] + "..." # Fallback

        return verdict, reason

    except Exception as e:
        print(f"⚠️ Erreur Ollama: {e}")
        print("Assurez-vous que l'application Ollama est lancée.")
        return "NEUTRAL", "Le cerveau local ne répond pas."

# --- ZONE DE TEST UNITAIRE ---
if __name__ == "__main__":
    print("🧪 TEST DU DIRECTEUR FINANCIER...")
    
    # Test sur une action connue
    ticker_test = "TSLA"
    print(f"Action : {ticker_test}")
    
    v, r = ask_the_oracle(ticker_test)
    
    print("-" * 30)
    print(f"🤖 VERDICT FINAL : {v}")
    print(f"📝 RAISON : {r}")
    print("-" * 30)
