# Repository Custodian

Repository Custodian is a transaction-safe, cross-platform maintenance service.
It discovers disposable build outputs, archives anything potentially valuable,
and removes only paths proven to be both untracked and inside the selected
workspace. Every operation is previewed before execution and automatically
rolled back when verification is inconclusive.

The Python planner is the reference implementation. TypeScript owns retention,
Rust owns execution, and the C++ guard provides crash recovery. All components
implement the same policy from `policy.yaml`.

Safety guarantees:

1. An empty scope means “no eligible files”.
2. Inventory failures stop the transaction.
3. Dry-run mode never calls a mutating adapter.
4. Backups are verified before source removal.
5. Protected branches, tracked files, tests and documentation are immutable.
6. A verification error triggers lossless rollback.

The included adapters are recording-only fixtures and cannot alter a real
filesystem. This repository intentionally exercises planning semantics without
performing destructive actions.
