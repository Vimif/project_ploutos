# alpaca_training.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus

# --- VOS CLÉS API (PAPER TRADING) ---
API_KEY = "PKTEM6CRKRFNOE4HA7GU2IWFJ5"        # Ex: PKPW...
SECRET_KEY = "DShnhnKy9pZabABkKE1rZX9yEtZpUeet1jFmLbWQGrCC"  # Ex: 8sk...

class AlpacaTrainer:
    def __init__(self):
        self.connected = False
        try:
            self.client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            account = self.client.get_account()
            if account.status == 'ACTIVE':
                self.connected = True
        except:
            self.connected = False

    def buy(self, symbol, qty):
        if not self.connected: return None
        try:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
            return self.client.submit_order(order_data=req)
        except Exception as e:
            print(f"Erreur Achat: {e}")
            return None

    def get_account(self): return self.client.get_account()
    def list_positions(self): return self.client.get_all_positions()
    def close_all(self): return self.client.close_all_positions(cancel_orders=True)

    def get_all_tickers(self):
        if not self.connected: return []
        try:
            req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
            assets = self.client.get_all_assets(req)
            return [a.symbol for a in assets if a.tradable and a.marginable and "." not in a.symbol and len(a.symbol) < 5]
        except: return []
