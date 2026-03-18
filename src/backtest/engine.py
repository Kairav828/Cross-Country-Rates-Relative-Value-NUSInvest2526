import pandas as pd
from src.backtest.cost_model import get_roundtrip_cost_bps, apply_cost_to_pnl

def run_backtest(signals_df: pd.DataFrame, cost_bps_per_leg: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts signals into a ledger of trades and an equity curve, strictly avoiding lookahead bias.
    """
    trades = []
    
    in_position = False
    current_direction = 0
    entry_time = None
    entry_spread = None
    
    if signals_df.index.name != 'date':
        signals_df = signals_df.copy()
        signals_df.index = pd.to_datetime(signals_df.index)

    roundtrip_cost_bps = get_roundtrip_cost_bps(cost_bps_per_leg)

    signals_df = signals_df.copy()
    signals_df['exec_target'] = signals_df['target_position'].shift(1, fill_value=0)
    signals_df['exec_exit'] = signals_df['exit_signal'].shift(1, fill_value=False)

    for current_time, row in signals_df.iterrows():
        target_pos = row["exec_target"]
        exit_signal = row["exec_exit"]
        current_spread = row["spread_t"]

        # 1. Check for Exits
        if in_position and (exit_signal or (target_pos != 0 and target_pos != current_direction)):
            gross_pnl = current_direction * (current_spread - entry_spread)
            net_pnl = apply_cost_to_pnl(gross_pnl, roundtrip_cost_bps)
            holding_days = (current_time - entry_time).days

            trades.append({
                "entry_time": entry_time,
                "exit_time": current_time,
                "direction": current_direction,
                "entry_spread": entry_spread,
                "exit_spread": current_spread,
                "pnl": net_pnl,
                "holding_days": holding_days
            })
            
            in_position = False
            current_direction = 0
            entry_time = None
            entry_spread = None

        # 2. Check for Entries
        if not in_position and target_pos != 0:
            in_position = True
            current_direction = target_pos
            entry_time = current_time
            entry_spread = current_spread

    trades_df = pd.DataFrame(trades)
    
    # 3. Build Equity Curve
    equity_df = pd.DataFrame(index=signals_df.index)
    equity_df["daily_pnl"] = 0.0
    
    if not trades_df.empty:
        pnl_by_date = trades_df.groupby("exit_time")["pnl"].sum()
        equity_df["daily_pnl"] = equity_df.index.map(pnl_by_date).fillna(0.0)
    
    equity_df["equity_curve"] = equity_df["daily_pnl"].cumsum()

    return trades_df, equity_df