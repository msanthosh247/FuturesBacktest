import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

class Trade:
    def __init__(
        self,
        open_idx: int,
        qty: int,
        entry_price: float,
        direction: int,
        commission: float,
        initial_margin: float,
        entry_time: pd.Timestamp,
    ):
        self.open_idx = open_idx
        self.qty = qty
        self.entry_price = entry_price
        self.direction = direction
        self.commission = commission
        self.initial_margin = initial_margin
        self.current_margin = initial_margin  # Current margin starts as initial margin
        self.entry_time = entry_time
        self.exit_time: Optional[pd.Timestamp] = None  # Exit time is set when the trade is closed
    
    def __repr__(self):
        return f"""
                open_idx : {self.open_idx}
                qty : {self.qty}
                entry_price : {self.entry_price}
                direction : {self.direction}
                commission : {self.commission}
                initial_margin : {self.initial_margin}
                current_margin : {self.current_margin}
                entry_time : {self.entry_time}
                """


class FuturesPortfolio:
    def __init__(
        self,
        initial_capital: float,
        margin: float,
        maintenance_margin_ratio: float,
        slippage: float,
        multiplier: int,
        default_qty: int,
        currentposition : int = 0 , 
    ):
        self.initial_capital = initial_capital
        self.margin = margin
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.slippage = slippage
        self.multiplier = multiplier
        self.default_qty = default_qty
        self.cash = initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Dict] = []
        self.casharray = []
        self.currPosition = currentposition

    def get_execution_price(self, current_price: float, is_buy: bool) -> float:
        """
        Calculate the execution price based on slippage.
        """
        if is_buy:
            return current_price * (1 + self.slippage)
        else:
            return current_price * (1 - self.slippage)

    def calculate_buy_qty(self, buy_price: float) -> int:
        """
        Calculate the quantity to buy based on available cash and margin requirements.
        """
        margin_req = buy_price * self.multiplier * self.margin # per contract
        buy_qty = int(self.cash / margin_req)
        buy_qty = min(buy_qty, self.default_qty) # margin_req - 1000 , default_qty = 2 , cash = 1050
        return buy_qty

    def open_trade(self, idx: int, price: float, is_long: bool, timestamp: pd.Timestamp) -> None:
        """
        Open a new trade.
        """
        buy_price = self.get_execution_price(price, is_long)
        buy_qty = self.calculate_buy_qty(buy_price)
        
        if buy_qty > 0:
            self.currPosition += 1 if is_long else -1
            commission = buy_qty * 0.57  # Fixed commission rate
            initial_margin = buy_price * self.margin * buy_qty * self.multiplier
            trade = Trade(idx, buy_qty, buy_price, 1 if is_long else -1, commission, initial_margin, timestamp)
            self.open_trades.append(trade)
            self.cash -= initial_margin
            self.cash -= commission
            return 1
        else:
            return -1

    def close_trade(self, idx: int, price: float, trade: Trade, timestamp: pd.Timestamp) -> None:
        """
        Close an existing trade.
        """
        sell_price = self.get_execution_price(price, trade.direction == -1)
        commission = trade.qty * 0.57  # Fixed commission rate
        self.cash += trade.current_margin  # Return the current margin (including top-ups) # daily settlements
        self.cash -= commission
        pnl = (sell_price - trade.entry_price) * trade.qty * self.multiplier * trade.direction
        self.currPosition -= trade.direction

        self.cash += pnl
        trade.exit_time = timestamp  # Set the exit time
        self.closed_trades.append({
            'open_idx': trade.open_idx,
            'close_idx': idx,
            'entry_price': trade.entry_price,
            'exit_price': sell_price,
            'qty': trade.qty,
            'direction': trade.direction,
            'pnl': pnl,
            'commission': trade.commission + commission,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
        })

    def check_margin(self, idx: int, price: float) -> None:
        """
        Check if the current margin for each trade is above the maintenance margin.
        If not, top up the margin or close the trade.
        """
        for i , trade in enumerate(self.open_trades):
            # Calculate the required maintenance margin
            maintenance_margin = trade.initial_margin * self.maintenance_margin_ratio
            # curr margin is the margin balance
            curr_margin = trade.current_margin + (price - trade.entry_price)*trade.qty*trade.direction * self.multiplier
            # Check if the current margin is below the maintenance margin
            if curr_margin < maintenance_margin:
                margin_deficit = trade.initial_margin - curr_margin
                if self.cash >= margin_deficit:
                    # Top up the margin using portfolio cash
                    self.cash -= margin_deficit
                    trade.current_margin += margin_deficit  # Add to current margin
                    # print(f"Margin topped up for trade opened at index {trade.open_idx}. Deficit: {margin_deficit}")
                else:
                    # Margin call: Not enough cash to top up, close the position
                    print(f"MARGIN CALL: Closing trade opened at index {trade.open_idx} due to insufficient margin.")
                    self.close_trade(idx, price, trade, pd.Timestamp.now())  # Use current timestamp for exit time
                    # self.open_trades.remove(trade)
                    del self.open_trades[i]


            elif curr_margin > trade.initial_margin:
                self.cash += curr_margin - trade.initial_margin
                trade.current_margin -=  curr_margin - trade.initial_margin

                

    def update_portfolio(self, idx: int, price: float, position: int, timestamp: pd.Timestamp) -> None:
        """
        Update the portfolio by checking margins and adjusting open trades.
        """
        # Check margin for all open trades

        # Adjust open trades to match the desired position
        while  position != self.currPosition:
            isLong = position > self.currPosition
            if len(self.open_trades) > 0 :
                newTradeToBeOpened = ((self.open_trades[0].direction ==  1) and isLong)
                newTradeToBeOpened = newTradeToBeOpened or ((self.open_trades[0].direction ==  -1) and (not isLong))

                if newTradeToBeOpened:
                    if self.open_trade(idx, price, isLong, timestamp) < 0:
                        break
                else:
                    trade = self.open_trades.pop(0)
                    self.close_trade(idx, price, trade, timestamp)
            else:
                if self.open_trade(idx, price , isLong , timestamp) < 0 :
                    break

    def get_portfolio_value(self, idx: int, price: float) -> float:
        """
        Calculate the current portfolio value.
        """
        self.check_margin(idx, price)
        portfolio_value = self.cash
        for trade in self.open_trades:
            sell_price = self.get_execution_price(price, trade.direction == -1)
            pnl = (sell_price - trade.entry_price) * trade.qty * trade.direction * self.multiplier
            commission = trade.qty * 0.57
            portfolio_value += pnl + trade.current_margin - commission
        return portfolio_value


def calculate_portfolio(
    position_size: np.ndarray,
    ohlc: np.ndarray,
    timestamps: List[pd.Timestamp],
    initial_capital: float,
    margin: float, # 0.08 emini futures
    maintenance_margin_ratio: float, # 0.6 
    slippage: float,
    multiplier: int,
    default_qty: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate the portfolio values and trade sheet over time.
    """
    portfolio = FuturesPortfolio(
                                    initial_capital, 
                                    margin, 
                                    maintenance_margin_ratio, 
                                    slippage, 
                                    multiplier, 
                                    default_qty
                                )
    n = len(ohlc)
    portfolio_values = np.full(n, np.nan)
    portfolio_values[0] = initial_capital
    casharray =[]

    for i in range(0, n):
        if i >= 520:
            pass
        portfolio.update_portfolio(i, ohlc[i , 0], position_size[i], timestamps[i])
        portfolio_values[i] = portfolio.get_portfolio_value(i, ohlc[i, 3])
        casharray.append(portfolio.cash)
        

    trade_sheet = pd.DataFrame(portfolio.closed_trades)
    return portfolio_values, trade_sheet , casharray