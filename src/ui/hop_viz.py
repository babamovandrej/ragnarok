from src.agent.state import HopRecord


def render_hop_path(hop_path: list[HopRecord], repo_slug: str, repo_sha: str) -> str:
    """Render an ordered hop path as markdown with clickable GitHub citation links."""
    if not hop_path:
        return "_No hops were needed for this answer._"

    lines = [f"**Hop path** ({len(hop_path)} hop{'s' if len(hop_path) != 1 else ''}):"]
    for i, hop in enumerate(hop_path, start=1):
        span = f"{hop.file_path}#L{hop.start_line}-L{hop.end_line}"
        url = f"https://github.com/{repo_slug}/blob/{repo_sha}/{hop.file_path}"
        url += f"#L{hop.start_line}-L{hop.end_line}"
        lines.append(f"{i}. [`{hop.symbol_name}` ({span})]({url}) — {hop.reason}")
    return "\n".join(lines)
