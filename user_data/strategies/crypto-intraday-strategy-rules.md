# Crypto Intraday Mean-Reversion Strategy — Rules Spec (for implementation)

**Asset:** BTC/USDT, ETH/USDT (perp or spot, high liquidity only)
**Timeframe:** 5-minute candles, 24/7
**Direction:** Long and Short (symmetric)

---

## Parameters (all configurable, not hardcoded)

| Param | Default |
|---|---|
| Z_WINDOW | 20 candles |
| Z_ENTRY_THRESHOLD | 2.0 |
| HURST_WINDOW | 100 candles |
| HURST_MAX | 0.45 |
| ATR_WINDOW | 14 |
| ATR_PCTL_WINDOW | 100 candles |
| ATR_PCTL_MIN | 30 |
| ATR_PCTL_MAX | 75 |
| VOL_WINDOW | 20 candles |
| VOLUME_Z_THRESHOLD | 1.5 |
| SL_ATR_MULT | 1.2 |
| TP_R_MULT | 2.0 |
| RISK_PCT_PER_TRADE | 0.5% |

---

## Indicator Calculations

```
μ  = mean(Close, Z_WINDOW)
σ  = stdev(Close, Z_WINDOW)
Z  = (Close_t - μ) / σ

H  = Hurst exponent via R/S analysis over HURST_WINDOW

ATR = ATR(ATR_WINDOW)
ATR_percentile = percentile_rank(ATR_t, trailing ATR_PCTL_WINDOW)

Volume_mean = mean(Volume, VOL_WINDOW)
Volume_std  = stdev(Volume, VOL_WINDOW)
Volume_Z    = (Volume_t - Volume_mean) / Volume_std
```

---

## Entry Rules

### LONG — all conditions must hold on same candle:
```
Z <= -Z_ENTRY_THRESHOLD
H < HURST_MAX
ATR_PCTL_MIN <= ATR_percentile <= ATR_PCTL_MAX
Volume_Z >= VOLUME_Z_THRESHOLD
```
Entry = candle close (or next candle open)

### SHORT — all conditions must hold on same candle:
```
Z >= Z_ENTRY_THRESHOLD
H < HURST_MAX
ATR_PCTL_MIN <= ATR_percentile <= ATR_PCTL_MAX
Volume_Z >= VOLUME_Z_THRESHOLD
```
Entry = candle close (or next candle open)

---

## Stop Loss

```
LONG:  SL = Entry - (SL_ATR_MULT * ATR)
SHORT: SL = Entry + (SL_ATR_MULT * ATR)
```

---

## Take Profit

```
LONG:
  TP_meanrevert = μ
  TP_Rmultiple  = Entry + (TP_R_MULT * (Entry - SL))
  TP = MIN(TP_meanrevert, TP_Rmultiple)

SHORT:
  TP_meanrevert = μ
  TP_Rmultiple  = Entry - (TP_R_MULT * (SL - Entry))
  TP = MAX(TP_meanrevert, TP_Rmultiple)
```

---

## Position Sizing

```
Risk_amount = Capital * RISK_PCT_PER_TRADE
Qty = Risk_amount / ABS(Entry - SL)
```

---

## Exit Logic (in priority order, check every candle after entry)

1. If price hits SL → exit full position, loss = -1R
2. If price hits TP → exit full position, profit booked
3. Time stop: if neither hit within 45 minutes (9 candles) AND price hasn't moved beyond 0.5R in favor → exit at market
4. No overnight/multi-day carry — this is intraday; define a hard session-end flatten rule if deploying on a venue/account with funding or session boundaries

---

## Cost Model (apply per trade, both legs)

```
Final_PnL = Gross_PnL - (Entry_fee + Exit_fee + Funding_cost_if_perp + Slippage_estimate)
```
- Use actual exchange maker/taker fee schedule
- If using perpetual futures, include funding rate accrual for time in position
- Model slippage separately from fees — do not assume candle-close fills in backtest

---

## Non-Negotiable Implementation Notes

- Recalculate Z, H, ATR_percentile, Volume_Z fresh on every new candle — never cache
- Do not assume Gaussian returns — validate empirical distribution of Z at entry threshold before trusting the 2.0 default; crypto often needs 2.5–3.0 due to fat tails
- Hurst exponent must be computed on a rolling window, not once — regime shifts intraday
- Backtest with realistic tick/orderbook-level fills, not candle-close fills
- Minimum ~30 trades per regime (ranging vs trending period) before treating backtest results as statistically meaningful
