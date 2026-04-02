export type PermissionLevel =
  | "read"
  | "file_write"
  | "internet"
  | "financial";

export type RiskLevel = "low" | "medium" | "high";

export interface TaskRequest {
  id: string;
  title: string;
  description: string;
  scheduledAt: string;
  permissions: PermissionLevel[];
  riskLevel: RiskLevel;
  pluginId: string;
  payload: Record<string, unknown>;
}

export interface UserPolicy {
  requireApprovalFor: PermissionLevel[];
  allowNightExecution: boolean;
  killSwitchEnabled: boolean;
  financial: {
    enabled: boolean;
    dailyLossLimitUsd: number;
    maxOpenPositions: number;
    nightlyTradesLimit: number;
  };
}

export interface PolicyDecision {
  approved: boolean;
  reasons: string[];
  requiresApproval: boolean;
}

export interface ExecutionContext {
  now: Date;
  policy: UserPolicy;
}

export interface ExecutionResult {
  success: boolean;
  summary: string;
  details?: Record<string, unknown>;
}

export interface AgentPlugin {
  id: string;
  name: string;
  requiredPermissions: readonly PermissionLevel[];
  run(task: TaskRequest, context: ExecutionContext): Promise<ExecutionResult>;
}

export interface AuditEvent {
  timestamp: string;
  taskId: string;
  eventType:
    | "policy_decision"
    | "task_started"
    | "task_finished"
    | "task_failed"
    | "feedback_recorded";
  message: string;
  metadata?: Record<string, unknown>;
}

export interface UserFeedback {
  taskId: string;
  correct: boolean;
  note?: string;
}

export interface LearningState {
  pluginScores: Record<string, number>;
  totalFeedbackCount: number;
}
