import { PolicyDecision, TaskRequest, UserPolicy } from "./types";

export class PolicyEngine {
  evaluate(task: TaskRequest, policy: UserPolicy): PolicyDecision {
    const reasons: string[] = [];

    if (policy.killSwitchEnabled) {
      reasons.push("Kill switch is enabled.");
      return { approved: false, reasons, requiresApproval: false };
    }

    if (!policy.allowNightExecution) {
      const hour = new Date(task.scheduledAt).getHours();
      if (hour >= 22 || hour < 6) {
        reasons.push("Night execution is disabled by policy.");
      }
    }

    if (task.permissions.includes("financial") && !policy.financial.enabled) {
      reasons.push("Financial actions are disabled.");
    }

    const requiresApproval = task.permissions.some((permission) =>
      policy.requireApprovalFor.includes(permission)
    );

    return { approved: reasons.length === 0, reasons, requiresApproval };
  }
}
