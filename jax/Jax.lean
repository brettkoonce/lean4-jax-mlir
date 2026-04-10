-- Jax — Phase 2 backend.
-- Lean 4 → idiomatic JAX Python via metaprogramming. The generated script
-- is written to .lake/build/generated_*.py and run via python3.
--
-- Re-exports the core spec types from LeanMlir so JAX-only main files can
-- use a single `import Jax` instead of importing each piece.
import LeanMlir.Types
import LeanMlir.Spec
import Jax.Codegen
import Jax.Runner
