def get_roundtrip_cost_bps(bps_per_leg: float = 5.0) -> float:
    """
    Calculates total roundtrip transaction costs in basis points.
    Default assumption: 5 bps per leg = 10 bps roundtrip.
    """
    return bps_per_leg * 2

def apply_cost_to_pnl(gross_pnl: float, cost_bps: float = 10.0) -> float:
    """
    Deducts transaction costs from the gross PnL.
    Note: 10 bps is 0.0010 in absolute rate terms.
    """
    cost_in_rates = cost_bps / 10000.0
    return gross_pnl - cost_in_rates