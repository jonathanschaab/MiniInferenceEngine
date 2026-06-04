## Key Principles
* **Readability:** Code should be easy to understand for all team members.
* **Maintainability:** Code should be easy to modify and extend.
* **Consistency:** Adhering to a consistent style across all projects improves collaboration and reduces errors.
* **Performance:** While readability is paramount, code should be efficient.

## Formatting
* **`rustfmt`:** All code should be formatted using the standard `rustfmt` tool with its default settings. This ensures a consistent visual style across the entire project.
* **Indentation:** Use 4 spaces per indentation level.

## Naming Conventions
* **Variables and Functions:** Use lowercase with underscores (`snake_case`): `let user_name`, `fn calculate_total()`
* **Constants:** Use uppercase with underscores (`UPPER_SNAKE_CASE`): `const MAX_CONNECTIONS: u32 = 100;`
* **Structs, Enums, and Traits:** Use `PascalCase`: `struct UserSession`, `enum ClientMode`
* **Modules:** Use `snake_case` for file and directory names: `client_handler.rs`, `server_logic/`

## Core Project Policies
These are fundamental rules that must be followed to ensure the quality, stability, and correctness of the server.

* **No `unsafe` Code:** The use of `unsafe` blocks is strictly prohibited. All code must be written in safe Rust.
* **No Panics:** With the exception of tests, the code must be robust and must not panic under any circumstances. Avoid using code that can panic, such as `unwrap()`, `expect()`, `unreachable!()`, or index operations that can go out of bounds. Always handle potential errors gracefully, typically by propagating a `Result`.
* **DRY (Don't Repeat Yourself):** Strive to keep the codebase as DRY as possible. Abstract and reuse common logic, data structures, and patterns to improve maintainability and reduce the chance of bugs.
* **Dependency Management:** Before adding a new dependency, check if a reasonable alternative already exists within the project's current dependency tree. The goal is to keep the number of dependencies minimal.
* **Keep Documentation Updated:** Any change to the code must be accompanied by corresponding changes to its documentation. Out-of-date documentation is a source of bugs.
* **Modern Rust:** This project uses Rust 1.96.0 which was released on 4/16/2026 and is stable. This means Let chains are stable as are if let guards. is_none_or() is also stable. use these new features when they improve the code.
* **No marketing:** Avoid flourishes and self praise in comments and documentation. Don't describe things as flawless, highly intelligent, or incredible. Also avoid unnecessary exclamation points.
* **Configuration Drift:** When adding new fields to `AppConfig`, you must also add them as commented-out default examples in `config.toml` to aid user discoverability.

## Testing
* **Test Coverage:** All new features and bug fixes must be accompanied by comprehensive tests. If you are fixing a bug that was not caught by existing tests, you must add a new test case that reproduces the bug to prevent regressions.
* **Use `expect()` in Tests:** While production code must not panic, tests can. However, use `expect("...")` with a clear explanation of why the panic is not expected in the test, rather than `unwrap()`.
* **Tests and Refactoring:** When refactoring code, do not delete or modify existing tests unless the underlying functionality is changing. The existing test suite should be used to validate the refactoring. Adding new tests to cover refactored code is encouraged.