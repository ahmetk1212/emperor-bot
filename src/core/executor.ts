import { AuditLog } from "./auditLog";
import { PluginRegistry } from "./pluginRegistry";
import {
  ExecutionContext,
  ExecutionResult,
  PolicyDecision,
  TaskRequest,
} from "./types";

export class TaskExecutor {
  constructor(
    private readonly plugins: PluginRegistry,
    private readonly audit: AuditLog
  ) {}

  async execute(
    task: TaskRequest,
    context: ExecutionContext,
    decision: PolicyDecision
  ): Promise<ExecutionResult> {
    if (!decision.approved) {
      return {
        success: false,
        summary: `Task blocked by policy: ${decision.reasons.join(" ")}`,
      };
    }

    const plugin = this.plugins.get(task.pluginId);
    if (!plugin) {
      return { success: false, summary: `Plugin not found: ${task.pluginId}` };
    }

    this.audit.add({
      timestamp: new Date().toISOString(),
      taskId: task.id,
      eventType: "task_started",
      message: `Started task ${task.title}`,
    });

    try {
      const result = await plugin.run(task, context);
      this.audit.add({
        timestamp: new Date().toISOString(),
        taskId: task.id,
        eventType: "task_finished",
        message: result.summary,
        metadata: result.details,
      });
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      this.audit.add({
        timestamp: new Date().toISOString(),
        taskId: task.id,
        eventType: "task_failed",
        message,
      });
      return { success: false, summary: message };
    }
  }
}
