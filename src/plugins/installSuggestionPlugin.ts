import { AgentPlugin, ExecutionContext, ExecutionResult, TaskRequest } from "../core/types";

export class InstallSuggestionPlugin implements AgentPlugin {
  readonly id = "install-suggestion";
  readonly name = "Install Suggestion Plugin";
  readonly requiredPermissions = ["internet"] as const;

  async run(task: TaskRequest, context: ExecutionContext): Promise<ExecutionResult> {
    const appName = String(task.payload.appName ?? "Unknown App");

    return {
      success: true,
      summary: `Installation suggestion prepared for ${appName}`,
      details: {
        generatedAt: context.now.toISOString(),
        recommendation: `Manual approval required before installing ${appName}.`,
      },
    };
  }
}
