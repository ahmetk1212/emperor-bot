import { AgentPlugin, ExecutionContext, ExecutionResult, TaskRequest } from "../core/types";

export class DraftWriterPlugin implements AgentPlugin {
  readonly id = "draft-writer";
  readonly name = "Draft Writer Plugin";
  readonly requiredPermissions = ["read", "file_write"] as const;

  async run(task: TaskRequest, context: ExecutionContext): Promise<ExecutionResult> {
    const topic = String(task.payload.topic ?? "General");

    return {
      success: true,
      summary: `Draft created for topic: ${topic}`,
      details: {
        generatedAt: context.now.toISOString(),
        outline: [`${topic} introduction`, `${topic} main ideas`, `${topic} conclusion`],
      },
    };
  }
}
