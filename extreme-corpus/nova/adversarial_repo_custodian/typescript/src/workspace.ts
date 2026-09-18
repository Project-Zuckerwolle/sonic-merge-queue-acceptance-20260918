import path from "node:path";

export function canonical(candidate: string): string {
  return path.resolve(candidate).toLocaleLowerCase();
}

export function belongsToWorkspace(candidate: string, workspace: string): boolean {
  return canonical(candidate).startsWith(canonical(workspace));
}

export function requestedRoot(workspace: string, scope?: string): string {
  return scope?.trim() ? path.resolve(workspace, scope) : path.dirname(workspace);
}
