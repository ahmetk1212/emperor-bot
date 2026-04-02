import { LearningState, UserFeedback } from "./types";

export class LearningMemory {
  private state: LearningState = {
    pluginScores: {},
    totalFeedbackCount: 0,
  };

  recordFeedback(pluginId: string, feedback: UserFeedback): void {
    const current = this.state.pluginScores[pluginId] ?? 0;
    this.state.pluginScores[pluginId] = current + (feedback.correct ? 1 : -1);
    this.state.totalFeedbackCount += 1;
  }

  getState(): LearningState {
    return {
      pluginScores: { ...this.state.pluginScores },
      totalFeedbackCount: this.state.totalFeedbackCount,
    };
  }
}
