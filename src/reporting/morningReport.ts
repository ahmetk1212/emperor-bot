import { AuditEvent } from "../core/types";

export interface MorningReport {
  generatedAt: string;
  totals: { started: number; finished: number; failed: number };
  highlights: string[];
}

export function generateMorningReport(events: AuditEvent[]): MorningReport {
  const started = events.filter((event) => event.eventType === "task_started").length;
  const finished = events.filter((event) => event.eventType === "task_finished").length;
  const failed = events.filter((event) => event.eventType === "task_failed").length;

  return {
    generatedAt: new Date().toISOString(),
    totals: { started, finished, failed },
    highlights: events.slice(-5).map((event) => `${event.eventType}: ${event.message}`),
  };
}
