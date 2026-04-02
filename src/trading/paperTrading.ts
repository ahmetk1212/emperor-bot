export interface PaperTradeRequest {
  symbol: string;
  side: "buy" | "sell";
  quantity: number;
  expectedPrice: number;
}

export interface TradingRiskPolicy {
  dailyLossLimitUsd: number;
  maxOpenPositions: number;
  nightlyTradesLimit: number;
}

export interface TradingState {
  openPositions: number;
  nightlyTradesExecuted: number;
  realizedPnlUsd: number;
}

export interface PaperTradeResult {
  accepted: boolean;
  reason: string;
  simulatedFillPrice?: number;
}

export class PaperTradingEngine {
  constructor(private readonly riskPolicy: TradingRiskPolicy) {}

  canExecute(request: PaperTradeRequest, state: TradingState): PaperTradeResult {
    if (state.openPositions >= this.riskPolicy.maxOpenPositions) {
      return { accepted: false, reason: "Max open positions limit reached." };
    }

    if (state.nightlyTradesExecuted >= this.riskPolicy.nightlyTradesLimit) {
      return { accepted: false, reason: "Nightly trade limit reached." };
    }

    if (state.realizedPnlUsd <= -Math.abs(this.riskPolicy.dailyLossLimitUsd)) {
      return { accepted: false, reason: "Daily loss limit exceeded." };
    }

    const slippage = request.side === "buy" ? 1.001 : 0.999;

    return {
      accepted: true,
      reason: "Paper trade accepted.",
      simulatedFillPrice: Number((request.expectedPrice * slippage).toFixed(4)),
    };
  }
}
