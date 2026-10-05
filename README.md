# Frontline — Ejyle AI Training

**Frontline** is the first line of support at a company selling Raspberry Pi hardware and
accessories. A ticket arrives; Frontline triages it and drafts a reply, publish to larger audience(customers)

## Recommended VScode settings:
### Extensions:
- Python
- Ruff
- Dependi

### settings.json(`ctrl + ,` to open)
```json
{
    // 1. Tell VS Code to use Pylance as the language server
    "python.languageServer": "Pylance",

    // 2. Reduce Pylance linting so Ruff can take over diagnostics
    "python.analysis.diagnosticMode": "openFilesOnly",
    
    // 3. Set up Pylance Type Checking (Off, Standard, or Strict)
    "python.analysis.typeCheckingMode": "standard",

    // 4. Python-specific editor automation settings
    "[python]": {
        // Set Ruff as the default formatter for Python files
        "editor.defaultFormatter": "charliermarsh.ruff", 
        "editor.formatOnSave": true,
        
        // Auto-fix errors and organize imports whenever you save
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    }
}
```
