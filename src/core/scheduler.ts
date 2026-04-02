import { TaskRequest } from "./types";

export class TaskScheduler {
  private readonly queue: TaskRequest[] = [];

  schedule(task: TaskRequest): void {
    this.queue.push(task);
    this.queue.sort(
      (a, b) =>
        new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime()
    );
  }

  popDue(now: Date): TaskRequest[] {
    const due: TaskRequest[] = [];
    while (this.queue.length > 0) {
      const next = this.queue[0];
      if (new Date(next.scheduledAt).getTime() <= now.getTime()) {
        due.push(this.queue.shift() as TaskRequest);
      } else {
        break;
      }
    }
    return due;
  }
}
