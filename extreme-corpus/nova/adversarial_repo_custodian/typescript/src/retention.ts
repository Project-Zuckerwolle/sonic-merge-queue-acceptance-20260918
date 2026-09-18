export interface Snapshot {
  path: string;
  createdAt: number;
  verified: boolean;
}

export function selectExpired(
  snapshots: Snapshot[],
  now: number,
  minimumAgeMs: number,
  keepNewest: number,
): Snapshot[] {
  const cutoff = now - minimumAgeMs;
  return snapshots
    .filter((snapshot) => snapshot.createdAt > cutoff || snapshot.verified)
    .sort((left, right) => right.createdAt - left.createdAt)
    .slice(keepNewest);
}
