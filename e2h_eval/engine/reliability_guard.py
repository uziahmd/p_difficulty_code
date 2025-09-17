# engine/reliability_guard.py
def reliability_guard():
    import builtins, os
    # Disable process/network/system-level escalation; allow basic FS ops so temp dirs can clean up
    for name in ("system", "popen", "fork", "spawnl", "spawnlp", "spawnv", "spawnve"):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("exit", "quit"):
        setattr(builtins, name, None)
    # (Optional) resource limits on Unix
    try:
        import resource
        # Address space ~1GB
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, hard))
        # CPU seconds 5s
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (5, hard))
        # Max files
        if hasattr(resource, "RLIMIT_NOFILE"):
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    except Exception:
        pass
