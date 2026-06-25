# Development principles

1. Keep each change small, reviewable, and testable.
2. After code changes, run the relevant tests before reporting completion.
3. When repository files change, verify the change, commit it with a clear message, and push it to the default remote branch before reporting completion, unless the user explicitly asks not to or credentials/remote access are unavailable.
4. Do not commit local automation, editor, machine, secret, or orchestration artifacts.
5. Commit messages should describe the product change, not the tool used to make it.

6. Prefer real-game integration over mock visuals when the user asks for StarCraft behavior; keep the mock only as a fast fallback/prototype.
