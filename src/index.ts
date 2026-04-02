import { AuditLog } from "./core/auditLog";
import { TaskExecutor } from "./core/executor";
import { LearningMemory } from "./core/memory";
import { PluginRegistry } from "./core/pluginRegistry";
import { PolicyEngine } from "./core/policyEngine";
import { TaskScheduler } from "./core/scheduler";
import { TaskRequest, UserPolicy } from "./core/types";
import { DraftWriterPlugin } from "./plugins/draftWriterPlugin";
import { InstallSuggestionPlugin } from "./plugins/installSuggestionPlugin";
import { generateMorningReport } from "./reporting/morningReport";

export interface AgentSystem {
  scheduler: TaskScheduler;
  policyEngine: PolicyEngine;
  executor: TaskExecutor;
  memory: LearningMemory;
  audit: AuditLog;
}

export function createAgentSystem(): AgentSystem {
  const registry = new PluginRegistry();
  registry.register(new DraftWriterPlugin());
  registry.register(new InstallSuggestionPlugin());

  const audit = new AuditLog();
  return {
    scheduler: new TaskScheduler(),
    policyEngine: new PolicyEngine(),
    executor: new TaskExecutor(registry, audit),
    memory: new LearningMemory(),
    audit,
  };
}

export async function runDueTasks(system: AgentSystem, policy: UserPolicy, now: Date): Promise<void> {
  const dueTasks = system.scheduler.popDue(now);

  for (const task of dueTasks) {
    const decision = system.policyEngine.evaluate(task, policy);
    system.audit.add({
      timestamp: now.toISOString(),
      taskId: task.id,
      eventType: "policy_decision",
      message: decision.approved ? "Approved" : "Blocked",
      metadata: { requiresApproval: decision.requiresApproval, reasons: decision.reasons },
    });

    await system.executor.execute(task, { now, policy }, decision);
  }
}

export function buildMorningSummary(system: AgentSystem): ReturnType<typeof generateMorningReport> {
  return generateMorningReport(system.audit.list());
}

export function sampleTask(): TaskRequest {
  return {
    id: "task-1",
    title: "Write nightly draft",
    description: "Prepare writing draft while user sleeps",
    scheduledAt: new Date().toISOString(),
    permissions: ["read", "file_write"],
    riskLevel: "low",
    pluginId: "draft-writer",
    payload: { topic: "Open-source local AI agents", language: "tr" },
  };
}
