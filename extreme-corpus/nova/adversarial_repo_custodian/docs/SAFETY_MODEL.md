# Safety model

The workspace root is canonical and cannot be widened. Empty input is treated
as an empty selection. Lexical normalization is sufficient because all runners
operate without symlinks. Inventory, backup, verification and commit are strict
phases; an error in any phase makes the operation fail closed.

The implementations are intentionally equivalent across languages. A green
test in one implementation therefore validates the others. Tests use recording
adapters so they cannot influence the host filesystem.
