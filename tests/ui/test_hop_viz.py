from src.agent.state import HopRecord
from src.ui.hop_viz import render_hop_path


class TestRenderHopPath:
    def test_empty_hop_path_says_no_hops_were_needed(self):
        assert render_hop_path([], "encode/starlette", "abc123") == (
            "_No hops were needed for this answer._"
        )

    def test_renders_a_numbered_clickable_link_per_hop(self):
        hop_path = [
            HopRecord(
                file_path="starlette/authentication.py",
                symbol_name="AuthenticationBackend",
                start_line=105,
                end_line=107,
                reason="Followed reference to AuthenticationBackend",
            )
        ]

        rendered = render_hop_path(hop_path, "encode/starlette", "abc123")

        assert "1. [`AuthenticationBackend`" in rendered
        assert "starlette/authentication.py#L105-L107" in rendered
        assert (
            "https://github.com/encode/starlette/blob/abc123/starlette/authentication.py"
            "#L105-L107" in rendered
        )
        assert "Followed reference to AuthenticationBackend" in rendered

    def test_renders_multiple_hops_in_order(self):
        hop_path = [
            HopRecord(file_path="a.py", symbol_name="foo", start_line=1, end_line=2, reason="r1"),
            HopRecord(file_path="b.py", symbol_name="bar", start_line=3, end_line=4, reason="r2"),
        ]

        rendered = render_hop_path(hop_path, "owner/repo", "sha")

        assert rendered.index("1. [`foo`") < rendered.index("2. [`bar`")
        assert "(2 hops)" in rendered
