"""Unit tests for StashProcessingBase class - pure logic methods."""

from datetime import UTC, datetime

import pytest
import respx

from download.types import DownloadType
from pathio import get_stash_path
from tests.fixtures.stash.stash_api_fixtures import dump_graphql_calls


class TestStashProcessingBase:
    """Test pure logic methods of StashProcessingBase class."""

    def test_generate_title_from_content(self, respx_stash_processor, faker):
        """Test _generate_title_from_content method - pure logic test."""
        # Test case 1: Content with a short first line
        content = "This is the title\nThis is the rest of the content"
        username = faker.user_name()
        created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Call method on real processor (inherits from StashProcessingBase)
        title = respx_stash_processor._generate_title_from_content(
            content, username, created_at
        )

        # Verify result
        assert title == "This is the title"

        # Test case 2: Content with a very long first line
        long_content = "X" * 200
        title = respx_stash_processor._generate_title_from_content(
            long_content, username, created_at
        )

        # Verify result is truncated with ellipsis
        assert title == ("X" * 125 + "...")
        assert len(title) == 128

        # Test case 3: No suitable content line, use fallback
        title = respx_stash_processor._generate_title_from_content(
            None, username, created_at
        )

        # Verify fallback format
        assert title == f"{username} - 2023/01/01"

        # Test case 4: With position indicators
        title = respx_stash_processor._generate_title_from_content(
            content, username, created_at, 2, 5
        )

        # Verify position is appended
        assert title == "This is the title - 2/5"

        # Test case 5: No content AND no created_at → username-only fallback.
        # created_at is Optional; a missing date must not crash the date format.
        title = respx_stash_processor._generate_title_from_content(None, username, None)
        assert title == username


class TestRequestTimeout:
    """Test _apply_request_timeout."""

    def test_timeout_applied_to_live_client(self, respx_stash_processor):
        """The configured timeout reaches the already-connected client.

        SGC hardcodes 30 s and its ``Timeout`` conn key is stripped by
        ``StashContext._normalize_conn_keys``, so a whole-creator flush (up to
        250 mutations in one request) died on a large creator. Both the gql
        execute timeout and the httpx transport must pick the new value up.
        """
        processor = respx_stash_processor
        processor.config.stash_request_timeout_s = 600

        processor._apply_request_timeout()

        client = processor.context.client
        assert client.gql_client.execute_timeout == 600
        assert client.http_transport.kwargs["timeout"] == 600
        assert client.http_transport.client.timeout.read == 600

    def test_missing_client_internals_warn_instead_of_raising(
        self, respx_stash_processor, monkeypatch
    ):
        """A library that renames these attributes must not break processing."""
        processor = respx_stash_processor
        monkeypatch.setattr(processor.context.client, "gql_client", None)
        monkeypatch.setattr(processor.context.client, "gql_ws_client", None)
        monkeypatch.setattr(processor.context.client, "http_transport", None)

        processor._apply_request_timeout()  # must not raise


class TestScanCreatorFolderBasePath:
    """base_path is the creator ROOT when scan_creator_folder has to create it."""

    @pytest.mark.asyncio
    async def test_created_base_path_is_creator_root_not_subfolder(
        self, respx_stash_processor, tmp_path, monkeypatch
    ):
        """A separated download type must not narrow the ownership root.

        With ``separate_timeline`` the created ``download_path`` is
        ``<creator>/Timeline``, but ``base_path`` is the root every ownership
        check compares against (``_image_files_all_local``,
        ``_sweep_creator_files``). Narrowing it to one subfolder makes every
        file outside that subfolder look foreign — no stamping, and a bogus
        "foreign co-resident" error per image.
        """
        processor = respx_stash_processor
        processor.config.download_directory = tmp_path
        processor.config.use_folder_suffix = True
        processor.config.separate_timeline = True
        processor.config.stash_scan_settle_s = 0
        processor.state.creator_name = "creator"
        processor.state.download_type = DownloadType.TIMELINE
        processor.state.base_path = None

        scanned: list[list[str]] = []

        async def fake_scan(paths, flags):
            scanned.append(paths)
            return "job-1"

        async def fake_wait(job_id):
            return True

        monkeypatch.setattr(processor.context.client, "metadata_scan", fake_scan)
        monkeypatch.setattr(processor.context.client, "wait_for_job", fake_wait)

        await processor.scan_creator_folder()

        creator_root = tmp_path / "creator_fansly"
        assert processor.state.download_path == creator_root / "Timeline"
        assert processor.state.base_path == creator_root
        # The whole creator folder is scanned, not just the Timeline subfolder.
        assert scanned == [[get_stash_path(creator_root, processor.config)]]


class TestPreloadEntities:
    """Test _preload_stash_entities."""

    @pytest.mark.asyncio
    async def test_preload_entities_makes_no_network_calls(self, respx_stash_processor):
        """_preload_stash_entities only configures TTLs — no GraphQL fetch.

        Mixin call sites use the ``store.filter() → store.find_one()``
        pattern, which lazy-populates the cache as entities are looked up.
        """
        route = respx.post("http://localhost:9999/graphql").mock(side_effect=[])

        try:
            await respx_stash_processor._preload_stash_entities()
        finally:
            dump_graphql_calls(
                route.calls, "test_preload_entities_makes_no_network_calls"
            )

        assert route.call_count == 0
