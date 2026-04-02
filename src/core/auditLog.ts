import { AuditEvent } from "./types";

export class AuditLog {
  private readonly events: AuditEvent[] = [];

  add(event: AuditEvent): void {
    this.events.push(event);
  }

  list(): AuditEvent[] {
    return [...this.events];
  }
}
