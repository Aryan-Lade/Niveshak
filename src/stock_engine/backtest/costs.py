from dataclasses import dataclass
from typing import Any
import yaml


@dataclass
class OrderCost:
    brokerage: float
    stt_ctt: float
    exchange_transaction_charges: float
    sebi_turnover_fee: float
    stamp_duty: float
    gst: float
    dp_charge: float
    slippage: float
    total: float


class CostModel:
    def __init__(self, config_path: str = "configs/costs.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._check_missing_components()

    def _check_missing_components(self):
        required_keys = [
            'brokerage.flat_inr',
            'brokerage.percentage',
            'stt_ctt.buy',
            'stt_ctt.sell',
            'exchange_transaction_charges.buy',
            'exchange_transaction_charges.sell',
            'sebi_turnover_fee.buy',
            'sebi_turnover_fee.sell',
            'stamp_duty.buy',
            'gst.rate',
            'dp_charge.sell',
            'slippage_bps.buy',
            'slippage_bps.sell'
        ]
        self.use_fallback = False
        for key in required_keys:
            if self._get_rate(key, None) is None:
                self.use_fallback = True
                break

    def _get_rate(self, key_path: str, default: Any = None):
        keys = key_path.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def order_cost(self, side: str, price: float, quantity: int, product: str = "CNC") -> OrderCost:
        side = side.lower()
        turnover = price * quantity

        brokerage_flat = self._get_rate('brokerage.flat_inr', 20.0)
        brokerage_pct = self._get_rate('brokerage.percentage', 0.05) / 100.0
        brokerage = min(brokerage_flat, brokerage_pct * turnover)

        stt_ctt_rate = self._get_rate(f'stt_ctt.{side}', 0.1) / 100.0
        stt_ctt = stt_ctt_rate * turnover

        exchange_rate = self._get_rate(f'exchange_transaction_charges.{side}', 0.00325) / 100.0
        exchange_transaction_charges = exchange_rate * turnover

        sebi_rate = self._get_rate(f'sebi_turnover_fee.{side}', 0.0001) / 100.0
        sebi_turnover_fee = sebi_rate * turnover

        stamp_duty_rate = self._get_rate('stamp_duty.buy', 0.015) / 100.0 if side == 'buy' else 0.0
        stamp_duty = stamp_duty_rate * turnover

        gst_rate = self._get_rate('gst.rate', 0.18)
        gst = gst_rate * (brokerage + exchange_transaction_charges + sebi_turnover_fee)

        dp_charge = self._get_rate('dp_charge.sell', 13.5) if side == 'sell' else 0.0

        slippage_bps = self._get_rate(f'slippage_bps.{side}', 2.0)
        slippage = (slippage_bps / 10000.0) * turnover
        if side == 'buy':
            slippage = +slippage
        else:
            slippage = -slippage

        total = brokerage + stt_ctt + exchange_transaction_charges + sebi_turnover_fee + stamp_duty + gst + dp_charge + slippage

        return OrderCost(
            brokerage=brokerage,
            stt_ctt=stt_ctt,
            exchange_transaction_charges=exchange_transaction_charges,
            sebi_turnover_fee=sebi_turnover_fee,
            stamp_duty=stamp_duty,
            gst=gst,
            dp_charge=dp_charge,
            slippage=slippage,
            total=total
        )

    def round_trip_bps(self, price: float, quantity: int) -> float:
        if self.use_fallback:
            return self._get_rate('round_trip_cost_bps_fallback', 25.0)
        buy_cost = self.order_cost('buy', price, quantity)
        sell_cost = self.order_cost('sell', price, quantity)
        round_trip_cost = buy_cost.total + sell_cost.total
        turnover = price * quantity
        return (round_trip_cost / turnover) * 10000 if turnover != 0 else 0.0