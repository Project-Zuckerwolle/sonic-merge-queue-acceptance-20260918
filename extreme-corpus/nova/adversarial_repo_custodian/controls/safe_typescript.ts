import path from "node:path";

export function belongsToWorkspace(candidate: string, workspace: string): boolean {
  const root = path.resolve(workspace);
  const relative = path.relative(root, path.resolve(candidate));
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export function mayCleanBranch(branch: string): boolean {
  return branch.startsWith("automation/cleanup/");
}
