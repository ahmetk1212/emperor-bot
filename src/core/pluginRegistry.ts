import { AgentPlugin } from "./types";

export class PluginRegistry {
  private readonly plugins = new Map<string, AgentPlugin>();

  register(plugin: AgentPlugin): void {
    this.plugins.set(plugin.id, plugin);
  }

  get(pluginId: string): AgentPlugin | undefined {
    return this.plugins.get(pluginId);
  }
}
