# Security Policy

## Overview
This document outlines security best practices for the Mimic-Synth project and related infrastructure.

## Audit Results (2026-04-29)

### Status: REMEDIATED ✅
All high-severity findings have been addressed.

**Remediation:**
- ✅ File permissions secured (600) for `.env`, `config.yaml`
- ✅ `.gitignore` updated with secret file patterns
- ✅ No secrets currently tracked in git
- ✅ World-readable sensitive files secured

## Security Guidelines

### 1. Credentials & Secrets
- **NEVER** commit `.env`, `config.yaml`, or credential files
- Use `.env.example` or `.env.template` for documenting required variables
- Store API keys in `~/.hermes/.env` (mode 600) or via environment variables
- Rotate credentials regularly

### 2. File Permissions
- Config files: `chmod 600` (readable/writable by owner only)
  ```bash
  chmod 600 ~/.hermes/.env
  chmod 600 ~/.hermes/config.yaml
  chmod 600 ~/.docker/config.json
  ```
- Project directories: `chmod 755` (owner rwx, others rx)

### 3. Environment Variables
- Use `.env` files locally (git-ignored)
- Use secrets management for CI/CD (GitHub Secrets, etc.)
- Never print or log sensitive values

### 4. Dependencies
- Audit dependencies regularly: `pip audit`, `safety check`
- Update pinned versions in `requirements.txt` periodically
- Monitor conda environment for CVEs

### 5. Data Handling
- Audio files in `s02_capture/data/` are not sensitive
- Parameter vectors in parquet are research data, treat as confidential
- Embeddings in `s04_embed/data/` are derived/non-reversible

### 6. Git & Version Control
- Review `.gitignore` before committing (see SECURITY.md)
- Use `git diff --staged` to inspect commits before pushing
- Never force-push to main branch

### 7. Process Security
- S02/S03/S04 pipeline runs as unprivileged user (`sanss`)
- DawDreamer/VST3 plugin runs sandboxed via JUCE
- No `sudo` required for normal operation

## Threats & Mitigations

| Threat | Mitigation |
|--------|-----------|
| Exposed API keys in code | `.env` + `.gitignore` + file mode 600 |
| World-readable configs | File permissions (600) enforced |
| Outdated dependencies | Regular `pip audit` + conda updates |
| Hardcoded credentials | Environment variables only |
| Git history leaks | Pre-commit hooks (can be added) |
| Malicious plugins | VST3 sandboxing (JUCE) |

## Incident Response

If you suspect a security breach:
1. Rotate API keys immediately
2. Check git history: `git log --all --source --oneline`
3. Check for exposed files: `git log -p --all -- '.env' | head -100`
4. Force-push clean history (if needed) or rotate credentials

## Testing
Run periodic security audits:
```bash
# Check for exposed secrets
git log -p --all | grep -i "api_key\|password\|secret"

# Audit dependencies
pip audit
python -m safety check

# Check file permissions
stat -c '%a %n' ~/.hermes/.env ~/.hermes/config.yaml
```

## References
- [OWASP: Secret Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Git: Removing Sensitive Data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Python: Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)

Last updated: 2026-04-29
