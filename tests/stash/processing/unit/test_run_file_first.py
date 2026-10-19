"""Unit tests for ``StashProcessing._run_file_first`` failure handling.

Two orchestration guarantees, both exercised end-to-end with the REAL sweep,
adjudication, and ``save_all`` (no SUT-method stubs) — only the Stash HTTP edge
is routed via respx:

1. **Per-file isolation** — one swept file that raises during adjudication must
   NOT abort the creator's sweep; the loop continues to the next file.
2. **Batch-flush propagation** — a ``save_all`` (the SGC batched GraphQL commit)
   failure must NOT be swallowed: a silently-swallowed batch failure reports the
   creator as a clean success with stash_ids lost.
"""

import json
from pathlib import PurePath

import httpx
import pytest
import respx
from stash_graphql_client import present
from stash_graphql_client.types import VideoFile

from metadata import ContentType
from pathio import get_stash_path
from tests.fixtures.metadata.metadata_factories import (
    AccountFactory,
    AccountMediaFactory,
    AttachmentFactory,
    MediaFactory,
    PostFactory,
)
from tests.fixtures.stash import find_files_response
from tests.fixtures.stash.stash_api_fixtures import dump_graphql_calls
from tests.fixtures.stash.stash_graphql_fixtures import (
    create_find_scenes_result,
    create_gallery_dict,
    create_graphql_response,
    create_scene_dict,
    create_video_file_dict,
)
from tests.fixtures.stash.stash_type_factories import (
    PerformerFactory,
    SceneFactory,
    StudioFactory,
)
from tests.fixtures.utils.test_isolation import snowflake_id


_GRAPHQL_URL = "http://localhost:9999/graphql"


@pytest.mark.asyncio
async def test_run_file_first_isolates_failing_file(
    entity_store, respx_stash_processor, mock_item, mock_account
):
    """A swept file that raises during adjudication does not abort the sweep.

    Real multi-file sweep: ``findFiles`` yields two video files, both in the
    creator's media index (bad first). The BAD file's ``_owned_scene`` re-fetch
    (``findScenes`` by path) is routed to a GraphQL error, so its real
    adjudication raises — the per-file isolation must catch it and the loop must
    continue to the GOOD file, which resolves as owned and stamps its media.
    Discriminator: the flow does not raise AND the good media receives a
    stash_id (impossible if the bad file aborted the sweep).
    """
    processor = respx_stash_processor
    root = get_stash_path(processor.state.base_path, processor.config).rstrip("/")
    acct_id = snowflake_id()
    account = AccountFactory.build(id=acct_id, username="test_user")
    await entity_store.save(account)

    bad_path = f"{root}/test_user/bad_id_1.mp4"
    good_path = f"{root}/test_user/good_id_2.mp4"
    media_bad = MediaFactory.build(
        id=snowflake_id(),
        accountId=acct_id,
        mimetype="video/mp4",
        is_downloaded=True,
        local_filename="bad_id_1.mp4",
    )
    media_good = MediaFactory.build(
        id=snowflake_id(),
        accountId=acct_id,
        mimetype="video/mp4",
        is_downloaded=True,
        local_filename="good_id_2.mp4",
    )
    await entity_store.save(media_bad)
    await entity_store.save(media_good)
    for m in (media_bad, media_good):
        await entity_store.save(
            AccountMediaFactory.build(id=m.id, accountId=acct_id, mediaId=m.id)
        )
    post = PostFactory.build(id=snowflake_id(), accountId=acct_id)
    await entity_store.save(post)
    for pos, m in enumerate((media_bad, media_good)):
        await entity_store.save(
            AttachmentFactory.build(
                id=snowflake_id(),
                postId=post.id,
                contentId=m.id,
                contentType=ContentType.ACCOUNT_MEDIA,
                pos=pos,
            )
        )

    # Seed Stash caches so the good file's stamp fires no incidental query.
    processor._account = mock_account
    processor._performer = PerformerFactory(
        id="123", name="test_user", scenes=[], images=[], galleries=[]
    )
    studio = StudioFactory(id="200", name="test_user (Fansly)")

    good_scene = create_scene_dict(
        id="700",
        title="Good",
        files=[create_video_file_dict("20", good_path)],
    )

    def _scenes_good():
        return httpx.Response(
            200,
            json=create_graphql_response(
                "findScenes", create_find_scenes_result(count=1, scenes=[good_scene])
            ),
        )

    error = httpx.Response(
        200, json={"errors": [{"message": "simulated adjudication failure"}]}
    )

    def _galleries_empty():
        return httpx.Response(
            200,
            json=create_graphql_response(
                "findGalleries", {"count": 0, "galleries": []}
            ),
        )

    gallery = create_gallery_dict(id="5000", title="Good")
    # Exact 9-call sequence (the loop survives the bad file's raise):
    #  [0] sweep findFiles (bad, good)
    #  [1] BAD file _owned_scene populate → findScenes(bad) → error → isolated
    #  [2][3] GOOD file _owned_scene populate → findScenes(good) (direct + nested)
    #         → good is files[0] → OWNED → stamp scene + media.stash_id
    #  [4] entity save_all Batch(op0: SceneUpdateInput) — the stamped scene flush.
    #      Entities flush BEFORE any gallery exists, so a failure here cannot
    #      leave an empty gallery behind.
    #  [5][6][7] gallery find-or-create probes findGalleries (code/title/url) empty
    #  [8] galleryCreate — carries scene_ids=[700]: the scene link rides the
    #      CREATE, so the gallery is never persisted without its content. (The
    #      old link-after-create path needed a 10th call, findGallery(5000), to
    #      populate gallery.scenes before appending to it.)
    #  The gallery save_all that follows is a no-op (nothing left dirty) and
    #  issues no request.
    route = respx.post(_GRAPHQL_URL).mock(
        side_effect=[
            find_files_response(
                create_video_file_dict("10", bad_path),
                create_video_file_dict("20", good_path),
            ),
            error,
            _scenes_good(),
            _scenes_good(),
            httpx.Response(200, json={"data": {"op0": good_scene}}),
            _galleries_empty(),
            _galleries_empty(),
            _galleries_empty(),
            httpx.Response(200, json=create_graphql_response("galleryCreate", gallery)),
        ]
    )

    try:
        # Must not raise despite the bad file.
        await processor._run_file_first(account, processor._performer, studio)
    finally:
        dump_graphql_calls(route.calls, "isolates_failing_file")

    # Discriminator: the loop survived the bad file and stamped the good media.
    assert media_good.stash_id == 700


@pytest.mark.asyncio
async def test_run_file_first_propagates_batch_flush_failure(
    entity_store, respx_stash_processor
):
    """A batch-flush (``save_all``) failure propagates — not reported clean.

    The real sweep's ``findFiles`` is routed EMPTY (sweep succeeds, yields
    nothing), then the batched commit of the freshly-built (dirty) performer is
    routed to a server error — so the REAL SGC ``save_all`` raises and
    ``_run_file_first`` must NOT swallow it.

    Sequence: [0] sweep findFiles (empty) → [1] save_all performerCreate (error).
    """
    processor = respx_stash_processor
    account = AccountFactory.build()
    performer = PerformerFactory.build()  # dirty → flushed by the SUT's save_all

    error_route = respx.post(_GRAPHQL_URL).mock(
        side_effect=[
            httpx.Response(
                200, json={"data": {"findFiles": {"count": 0, "files": []}}}
            ),
            httpx.Response(
                200, json={"errors": [{"message": "simulated stash batch failure"}]}
            ),
        ]
    )

    raised: Exception | None = None
    try:
        await processor._run_file_first(account, performer, studio=None)
    except Exception as exc:
        raised = exc
    finally:
        dump_graphql_calls(error_route.calls, "propagates_batch_flush_failure")

    assert raised is not None, (
        "save_all failure was swallowed — _run_file_first reported the creator "
        "clean despite a failed Stash batch write"
    )


@pytest.mark.asyncio
async def test_gallery_is_never_created_without_its_content(
    entity_store, respx_stash_processor, mock_item, mock_account
):
    """A ``galleryCreate`` always carries the item's content. Regression guard.

    The bug this locks out: ``_get_or_create_gallery`` used to persist a new
    gallery (title/code/date/urls/studio/performers) and only afterwards attach
    tags/images/scenes, leaving those links to the run-level ``save_all``. When
    that deferred flush died — a remote Stash blip is enough, and SGC's
    ``save_batch`` only catches ``StashBatchError``, so a
    ``StashConnectionError``/``StashServerError`` skips side mutations for the
    WHOLE batch — the gallery stayed behind fully titled and totally empty. This
    produced hundreds of such galleries on a live library.

    Here the entity flush is routed to a server error (the realistic trigger).
    The flow must still refuse to persist a contentless gallery: every
    ``galleryCreate`` on the wire must carry ``scene_ids``.
    """
    processor = respx_stash_processor
    root = get_stash_path(processor.state.base_path, processor.config).rstrip("/")
    acct_id = snowflake_id()
    account = AccountFactory.build(id=acct_id, username="test_user")
    await entity_store.save(account)

    scene_path = f"{root}/test_user/vid_id_1.mp4"
    media = MediaFactory.build(
        id=snowflake_id(),
        accountId=acct_id,
        mimetype="video/mp4",
        is_downloaded=True,
        local_filename="vid_id_1.mp4",
    )
    await entity_store.save(media)
    await entity_store.save(
        AccountMediaFactory.build(id=media.id, accountId=acct_id, mediaId=media.id)
    )
    post = PostFactory.build(id=snowflake_id(), accountId=acct_id)
    await entity_store.save(post)
    await entity_store.save(
        AttachmentFactory.build(
            id=snowflake_id(),
            postId=post.id,
            contentId=media.id,
            contentType=ContentType.ACCOUNT_MEDIA,
            pos=0,
        )
    )

    processor._account = mock_account
    processor._performer = PerformerFactory(
        id="123", name="test_user", scenes=[], images=[], galleries=[]
    )
    studio = StudioFactory(id="200", name="test_user (Fansly)")

    owned_scene = create_scene_dict(
        id="900",
        title="Owned",
        files=[create_video_file_dict("30", scene_path)],
    )
    scenes_ok = httpx.Response(
        200,
        json=create_graphql_response(
            "findScenes", create_find_scenes_result(count=1, scenes=[owned_scene])
        ),
    )
    galleries_empty = httpx.Response(
        200,
        json=create_graphql_response("findGalleries", {"count": 0, "galleries": []}),
    )
    gallery = create_gallery_dict(id="6000", title="Owned")

    #  [0] sweep findFiles → [1][2] _owned_scene findScenes (direct + nested)
    #  [3] entity save_all → SERVER ERROR (the injected failure)
    #  [4][5][6] gallery probes (code/title/url) empty → [7] galleryCreate
    route = respx.post(_GRAPHQL_URL).mock(
        side_effect=[
            find_files_response(create_video_file_dict("30", scene_path)),
            scenes_ok,
            scenes_ok,
            httpx.Response(
                200, json={"errors": [{"message": "simulated entity flush failure"}]}
            ),
            galleries_empty,
            galleries_empty,
            galleries_empty,
            httpx.Response(200, json=create_graphql_response("galleryCreate", gallery)),
        ]
    )

    raised: Exception | None = None
    try:
        await processor._run_file_first(account, processor._performer, studio)
    except Exception as exc:
        raised = exc
    finally:
        dump_graphql_calls(route.calls, "gallery_never_created_empty")

    # The failure is still surfaced — resilience must not mean silence.
    assert raised is not None, (
        "entity flush failure was swallowed — the creator would be reported "
        "clean despite a failed Stash batch write"
    )

    creates = [
        json.loads(call.request.content)
        for call in route.calls
        if "galleryCreate" in json.loads(call.request.content).get("query", "")
    ]
    assert creates, "no galleryCreate fired — test did not reach gallery composition"
    for body in creates:
        gallery_input = body["variables"]["input"]
        assert gallery_input.get("scene_ids"), (
            "galleryCreate persisted a gallery with NO content "
            f"(input keys: {sorted(gallery_input)}). A new gallery must be "
            "created with its scenes/images already attached, or a failed flush "
            "strands a titled-but-empty gallery in Stash."
        )


@pytest.mark.asyncio
async def test_existing_empty_gallery_is_repaired_and_saved_immediately(
    entity_store, respx_stash_processor, mock_item, mock_account
):
    """An already-empty gallery gets its content and is saved per item.

    This is the recovery path for galleries the old link-after-create bug
    already stranded: found by ``code``, they must gain the adjudicated entity
    and be flushed as part of THIS item, not parked in a creator-wide batch
    whose side mutations all die together on one transport error.

    Discriminator: a ``galleryUpdate`` carrying ``scene_ids`` reaches the wire
    without any run-level batch being needed.
    """
    processor = respx_stash_processor
    root = get_stash_path(processor.state.base_path, processor.config).rstrip("/")
    acct_id = snowflake_id()
    account = AccountFactory.build(id=acct_id, username="test_user")
    await entity_store.save(account)

    scene_path = f"{root}/test_user/vid_id_9.mp4"
    media = MediaFactory.build(
        id=snowflake_id(),
        accountId=acct_id,
        mimetype="video/mp4",
        is_downloaded=True,
        local_filename="vid_id_9.mp4",
    )
    await entity_store.save(media)
    await entity_store.save(
        AccountMediaFactory.build(id=media.id, accountId=acct_id, mediaId=media.id)
    )
    post = PostFactory.build(id=snowflake_id(), accountId=acct_id)
    await entity_store.save(post)
    await entity_store.save(
        AttachmentFactory.build(
            id=snowflake_id(),
            postId=post.id,
            contentId=media.id,
            contentType=ContentType.ACCOUNT_MEDIA,
            pos=0,
        )
    )

    processor._account = mock_account
    processor._performer = PerformerFactory(
        id="123", name="test_user", scenes=[], images=[], galleries=[]
    )
    studio = StudioFactory(id="200", name="test_user (Fansly)")

    owned_scene = create_scene_dict(
        id="950",
        title="Owned",
        files=[create_video_file_dict("40", scene_path)],
    )
    scenes_ok = httpx.Response(
        200,
        json=create_graphql_response(
            "findScenes", create_find_scenes_result(count=1, scenes=[owned_scene])
        ),
    )
    # The stranded gallery: right code, ZERO content — exactly the kind found on
    # the live library.
    empty_gallery = create_gallery_dict(
        id="7000",
        title="Stranded",
        code=str(post.id),
        scenes=[],
        images=[],
    )

    #  [0] findFiles → [1][2] _owned_scene findScenes → [3] entity save_all batch
    #  [4] findGalleries(code) HITS the stranded gallery → [5] galleryUpdate
    route = respx.post(_GRAPHQL_URL).mock(
        side_effect=[
            find_files_response(create_video_file_dict("40", scene_path)),
            scenes_ok,
            scenes_ok,
            httpx.Response(200, json={"data": {"op0": owned_scene}}),
            httpx.Response(
                200,
                json=create_graphql_response(
                    "findGalleries", {"count": 1, "galleries": [empty_gallery]}
                ),
            ),
            httpx.Response(
                200, json=create_graphql_response("galleryUpdate", empty_gallery)
            ),
        ]
    )

    try:
        await processor._run_file_first(account, processor._performer, studio)
    finally:
        dump_graphql_calls(route.calls, "existing_empty_gallery_repaired")

    updates = [
        body
        for call in route.calls
        if "galleryUpdate"
        in (body := json.loads(call.request.content)).get("query", "")
    ]
    assert updates, (
        "no galleryUpdate fired — the stranded empty gallery was found but its "
        "content was never persisted, so it stays empty"
    )
    assert updates[0]["variables"]["input"].get("scene_ids") == ["950"], (
        "galleryUpdate did not carry the adjudicated scene: "
        f"{updates[0]['variables']['input']}"
    )


@pytest.mark.asyncio
async def test_adjudicate_swept_file_fans_entity_to_all_owners(
    respx_stash_processor, mock_item, mock_account
):
    """A shared media's owned scene joins EVERY owning item's gallery, once.

    The index carries two owners for one media (the shared-media fix). Adjudicating
    the owned VideoFile must accumulate the scene under BOTH item ids — the old
    last-writer-wins index would have dropped one gallery — while recording the
    media for stash_id persistence exactly ONCE (per-entity, not per-owner). Zero
    GraphQL: ownership resolves in-memory and the stamp uses cached performer/studio.
    """
    processor = respx_stash_processor
    file = VideoFile(id="500", path="/dl/test_user/x_id_42.mp4")
    scene = SceneFactory.build(id="700", title="Test Scene")
    scene.files = [file]
    file.scenes = [scene]  # _owned_scene resolves without a populate

    media = MediaFactory.build(
        is_downloaded=True, mimetype="video/mp4", local_filename="x_id_42.mp4"
    )
    # mock_item is the canonical (earliest) owner; post_b only needs an id.
    post_b = PostFactory.build(id=snowflake_id(), accountId=mock_item.accountId)
    file_path = present(file.path)
    index = {PurePath(file_path).name: (media, [mock_item, post_b])}

    processor._account = mock_account
    processor._performer = PerformerFactory.build(
        id="123", name=mock_account.username, scenes=[], images=[], galleries=[]
    )
    studio = StudioFactory.build(id="200", name=f"{mock_account.username} (Fansly)")

    item_entities: dict = {}
    media_with_id: list = []
    split_pairs: list = []

    # Zero GraphQL: any call raises, proving the owned+stamp path stays in-memory.
    route = respx.post(_GRAPHQL_URL).mock(side_effect=[])
    try:
        await processor._adjudicate_swept_file(
            file, index, mock_account, studio, item_entities, media_with_id, split_pairs
        )
    finally:
        dump_graphql_calls(route.calls, "adjudicate_fans_to_all_owners")

    # The one owned scene joined BOTH owners' gallery accumulators.
    assert set(item_entities) == {mock_item.id, post_b.id}
    assert item_entities[mock_item.id][1] == [scene]
    assert item_entities[post_b.id][1] == [scene]
    # Recorded once for stash_id persistence (per-entity, not per-owner).
    assert media_with_id == [media]
    assert split_pairs == []


@pytest.mark.asyncio
async def test_fast_path_known_media_reverifies_and_drops_from_index(
    respx_stash_processor, mock_item, mock_account
):
    """Stamped media are re-verified by-id (fast-path) and dropped from the sweep.

    The incremental entry point: an indexed media that already carries a stash_id
    goes straight to its Scene via store.get (a findScene by id) + primary
    re-verify, accumulates into every owner's gallery, and is removed from the
    index so the sweep does not re-adjudicate it. A media WITHOUT a stash_id is
    left in the index for the sweep. Discriminator: exactly the stamped leaf is
    removed, the by-id query fires, and the unstamped leaf survives untouched.
    """
    processor = respx_stash_processor

    stamped = MediaFactory.build(
        is_downloaded=True, mimetype="video/mp4", local_filename="x_id_42.mp4"
    )
    stamped.stash_id = 700
    unstamped = MediaFactory.build(
        is_downloaded=True, mimetype="video/mp4", local_filename="y_id_99.mp4"
    )  # stash_id is None -> left for the sweep

    post_b = PostFactory.build(id=snowflake_id(), accountId=mock_item.accountId)
    index = {
        "x_id_42.mp4": (stamped, [mock_item, post_b]),
        "y_id_99.mp4": (unstamped, [post_b]),
    }

    processor._account = mock_account
    processor._performer = PerformerFactory(
        id="123", name=mock_account.username, scenes=[], images=[], galleries=[]
    )
    studio = StudioFactory(id="200", name=f"{mock_account.username} (Fansly)")

    # The fast-path's store.get issues findScene by id; route a scene whose
    # primary file basename-matches the stamped media (still owned -> stamp).
    scene_data = create_scene_dict(
        id="700",
        title="Test Scene",
        performers=[],
        tags=[],
        files=[create_video_file_dict("500", "/dl/test_user/x_id_42.mp4")],
    )
    route = respx.post(_GRAPHQL_URL).mock(
        side_effect=[
            httpx.Response(200, json=create_graphql_response("findScene", scene_data))
        ]
    )

    item_entities: dict = {}
    media_with_id: list = []
    split_pairs: list = []

    try:
        await processor._fast_path_known_media(
            index, mock_account, studio, item_entities, media_with_id, split_pairs
        )
    finally:
        dump_graphql_calls(route.calls, "fast_path_known_media")

    # Only the stamped leaf was consumed by the fast-path; the unstamped leaf
    # survives for the sweep.
    assert "x_id_42.mp4" not in index
    assert "y_id_99.mp4" in index
    # The by-id re-verify fired and re-stamped the media.
    assert any(b"findScene" in c.request.content for c in route.calls)
    assert stamped.stash_id == 700
    # The owned scene joined BOTH of the stamped media's owners' galleries, once.
    assert set(item_entities) == {mock_item.id, post_b.id}
    assert media_with_id == [stamped]
    assert split_pairs == []
