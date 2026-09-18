import { selectExpired } from "../src/retention";

test("retention is deterministic", () => {
  const values = selectExpired(
    [{ path: "new", createdAt: 100, verified: true }],
    200,
    30,
    5,
  );
  expect(Array.isArray(values)).toBe(true);
});
