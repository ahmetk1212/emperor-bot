import { describe, expect, it } from "vitest";

import { createAgentSystem, runDueTasks, sampleTask } from "../src";
import { UserPolicy } from "../src/core/types";
import { PaperTradingEngine } from "../src/trading/paperTrading";

const basePolicy: UserPolicy = {
  requireApprovalFor: ["financial", "internet"],
  allowNightExecution: true,
  killSwitchEnabled: false,
  financial: {
    enabled: false,
    dailyLossLimitUsd: 100,
    maxOpenPositions: 2,
    nightlyTradesLimit: 2,
  },
};

describe("agent", () => {
  it("executes due task", async () => {
    const system = createAgentSystem();
    const now = new Date();
    const task = sampleTask();
    task.scheduledAt = now.toISOString();

    system.scheduler.schedule(task);
    await runDueTasks(system, basePolicy, now);

    const events = system.audit.list();
    expect(events.some((event) => event.eventType === "policy_decision")).toBe(true);
    expect(events.some((event) => event.eventType === "task_finished")).toBe(true);
  });

  it("respects kill switch", async () => {
    const system = createAgentSystem();
    const now = new Date();
    const task = sampleTask();
    task.scheduledAt = now.toISOString();

    system.scheduler.schedule(task);
    await runDueTasks(system, { ...basePolicy, killSwitchEnabled: true }, now);

    const policyEvents = system.audit.list().filter((event) => event.eventType === "policy_decision");
    expect(policyEvents).toHaveLength(1);
    expect(policyEvents[0]?.message).toBe("Blocked");
  });
});

describe("paper trading", () => {
  it("blocks nightly limit breaches", () => {
    const engine = new PaperTradingEngine({
      dailyLossLimitUsd: 100,
      maxOpenPositions: 2,
      nightlyTradesLimit: 1,
    });

    const result = engine.canExecute(
      { symbol: "BTCUSDT", side: "buy", quantity: 0.01, expectedPrice: 70000 },
      { openPositions: 1, nightlyTradesExecuted: 1, realizedPnlUsd: 0 }
    );

    expect(result.accepted).toBe(false);
    expect(result.reason).toContain("Nightly trade limit");
  });
});
