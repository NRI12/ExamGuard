from alembic.config import Config
from alembic.script import ScriptDirectory


def test_camera_sources_is_the_current_revision() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))
    revisions = list(scripts.walk_revisions())

    assert scripts.get_heads() == ["20260924_0002"]
    assert len(revisions) == 2
    assert revisions[0].revision == "20260924_0002"
    assert revisions[0].doc == "camera_sources"
    assert revisions[0].down_revision == "20260920_0001"
    assert revisions[1].revision == "20260920_0001"
    assert revisions[1].doc == "initial_schema"
    assert revisions[1].down_revision is None
