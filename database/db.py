# database/db.py
"""Gestion de la base de données"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host='localhost',
        database='ploutos',
        user='ploutos',
        password=os.getenv('DB_PASSWORD', 'MotDePasseSecurise123!')
    )

def log_trade(symbol, action, qty, price, amount, reason=''):
    """Enregistrer un trade"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO trades (symbol, action, quantity, price, amount, reason)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (symbol, action, qty, price, amount, reason))
    
    conn.commit()
    cur.close()
    conn.close()

def get_trade_history(days=30):
    """Récupérer l'historique des trades"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT * FROM trades 
        WHERE timestamp > NOW() - INTERVAL '%s days'
        ORDER BY timestamp DESC
    """, (days,))
    
    trades = cur.fetchall()
    cur.close()
    conn.close()
    
    return trades