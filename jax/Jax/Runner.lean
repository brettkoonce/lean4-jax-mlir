import LeanJax.Types
import LeanJax.Spec
import LeanJax.Codegen
/-! Runner: find Python, generate script, execute training. -/

def findPython : IO String := do
  let r ← IO.Process.output { cmd := "test", args := #["-f", ".venv/bin/python3"] }
  if r.exitCode == 0 then return ".venv/bin/python3"
  return "python3"

def runJax (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind) (dataDir scriptName : String) : IO Unit := do
  IO.println s!"Lean 4 → JAX  {spec.name}"
  IO.println s!"  arch:   {spec.archStr}"
  IO.println s!"  params: {spec.totalParams}"
  IO.println s!"  data:   {dataDir}"
  IO.println ""

  let code := JaxCodegen.generate spec cfg ds dataDir
  let scriptPath := ".lake/build/" ++ scriptName
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile scriptPath code
  IO.println s!"Generated: {scriptPath} ({code.length} chars)"
  IO.println "Running JAX training...\n"

  let python ← findPython
  let child ← IO.Process.spawn {
    cmd := python
    args := #[scriptPath]
    stdout := .piped
    stderr := .piped
    stdin  := .null
  }

  let stdout ← child.stdout.readToEnd
  IO.print stdout

  let stderr ← child.stderr.readToEnd
  let exitCode ← child.wait
  if exitCode != 0 then
    IO.eprintln s!"\nJAX process exited with code {exitCode}"
    IO.eprintln stderr
