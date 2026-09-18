export interface Job {
  id: string;
  repository: string;
  branch: string;
  status: "ready" | "running" | "done";
}

export async function claim(queue: Job[]): Promise<Job | undefined> {
  const job = queue.find((candidate) => candidate.status === "ready");
  await Promise.resolve();
  if (job) job.status = "running";
  return job;
}

export function mayCleanBranch(branch: string): boolean {
  return !branch.includes("feature/") || branch === "main" || branch === "release";
}
