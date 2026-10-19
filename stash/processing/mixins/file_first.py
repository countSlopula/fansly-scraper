"""File-first adjudication flow for StashProcessing.

Owns both lookup entry points feeding the shared adjudication:

- ``_run_file_first`` — the full-creator sweep (also fills ``media.stash_id``).
- ``_run_file_first_incremental`` — the monitoring daemon's sweep-free pass:
  scan just-downloaded files, locate each by basename, adjudicate, flush.

Both reuse ``_prepare_file_first`` (index + accumulators), ``_fast_path_known_media``
(re-verify already-stamped media by id), ``_safe_adjudicate`` (per-file isolation),
and ``_compose_and_flush`` (one gallery per item + a single batched ``save_all``).
"""

from __future__ import annotations

import traceback
from pathlib import Path, PurePath

from stash_graphql_client.types import (
    BaseFile,
    BasicFile,
    Gallery,
    GalleryChapter,
    GalleryFile,
    Image,
    ImageFile,
    Performer,
    Scene,
    Studio,
    UnsetType,
    VideoFile,
)

from helpers.rich_progress import get_progress_manager
from metadata import Account, Media, Message, Post
from metadata.models import get_store
from pathio import get_stash_path
from textio import print_error, print_info

from ...logging import debug_print
from ...logging import processing_logger as logger
from ..protocols import StashProcessingProtocol


class FileFirstProcessingMixin(StashProcessingProtocol):
    """File-first sweep + incremental adjudication, gallery composition, flush."""

    # Re-declare base-class attrs assigned in this mixin's method bodies —
    # without these, mypy infers definitions that conflict with
    # StashProcessingBase's declarations at class-composition time.
    _account: Account | None
    _performer: Performer | None
    _stash_parent_task: str | None

    async def continue_stash_processing(
        self,
        account: Account | None,
        performer: Performer | None,
    ) -> None:
        """Continue processing in background.

        Args:
            account: Account to process
            performer: Performer created from account
        """
        progress_mgr = get_progress_manager()

        try:
            if not account or not performer:
                raise ValueError("Missing account or performer data")
            # Validate performer type (library returns Pydantic objects directly)
            if not isinstance(performer, Performer):
                raise TypeError("performer must be a Stash Performer object")

            self._account = account
            self._performer = performer

            # Convert performer.id (str) for comparison with account.stash_id (int)
            if account.stash_id != int(performer.id):
                await self._update_account_stash_id(
                    account=account,
                    performer=performer,
                )

            # 2 phases: studio, file-first adjudication (sweep + split + galleries)
            performer_label = performer.name or "creator"
            with progress_mgr.session():
                self._stash_parent_task = progress_mgr.add_task(
                    name="stash_creator",
                    description=f"Stash: {performer_label}",
                    total=2,
                    show_elapsed=True,
                )

                # Process creator studio
                print_info("Processing creator Studio...")
                studio = await self.process_creator_studio(account=account)
                self._studio = studio
                progress_mgr.update_task(self._stash_parent_task, advance=1)

                # Sweep the creator's Stash files, adjudicate each, compose
                # galleries, and flush in one batched save_all.
                print_info("Processing creator content (file-first)...")
                await self._run_file_first(account, performer, studio)
                progress_mgr.update_task(self._stash_parent_task, advance=1)

        except Exception as e:
            print_error(f"Error in Stash processing: {e}")
            logger.exception("Error in Stash processing", exc_info=e)
            debug_print(
                {
                    "method": "StashProcessing - continue_stash_processing",
                    "status": "processing_failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        finally:
            self._finalize_creator(performer)

    async def process_creator_incremental(self) -> None:
        """Daemon entry: awaited, sweep-free incremental Stash pass for one creator.

        Connects, resolves the studio + performer, then runs the incremental
        file-first flow (scan just-downloaded files -> find_one by basename ->
        adjudicate -> batched flush). Awaited inline so the daemon worker only
        marks the creator processed after Stash finishes.
        """
        if not await self._connect_stash():
            return
        account, performer = await self.process_creator()
        performer_obj = performer if isinstance(performer, Performer) else None
        try:
            if not account or not performer_obj:
                return
            self._account = account
            self._performer = performer_obj
            if account.stash_id != int(performer_obj.id):
                await self._update_account_stash_id(
                    account=account, performer=performer_obj
                )
            print_info("Processing creator Studio...")
            studio = await self.process_creator_studio(account=account)
            self._studio = studio
            print_info("Processing creator content (incremental)...")
            await self._run_file_first_incremental(account, performer_obj, studio)
        except Exception as e:
            print_error(f"Error in incremental Stash processing: {e}")
            logger.exception("Error in incremental Stash processing", exc_info=e)
            raise
        finally:
            self._finalize_creator(performer_obj)

    def _finalize_creator(self, performer: Performer | None) -> None:
        """Reset per-creator state, invalidate cached file types, log cache stats."""
        self._stash_parent_task = None
        self._account = None
        self._performer = None
        self._studio = None
        # invalidate_type is exact-match on __type_name__ (no subtype cascade),
        # so each concrete BaseFile subtype the sweep may have cached is listed.
        for entity_type in (
            Gallery,
            GalleryChapter,
            Scene,
            Image,
            VideoFile,
            ImageFile,
            GalleryFile,
            BasicFile,
        ):
            self.store.invalidate_type(entity_type)

        performer_name = (
            performer.name if isinstance(performer, Performer) else repr(performer)
        )
        stats = self.store.cache_stats()
        by_type = ", ".join(f"{k}={v}" for k, v in sorted(stats.by_type.items()))
        print_info(
            f"Finished Stash processing for {performer_name} "
            f"(cache: {stats.total_entries} entries — {by_type})"
        )

    async def _run_file_first(
        self,
        account: Account,
        performer: Performer,
        studio: Studio | None,
    ) -> None:
        """Run the file-first Stash processing flow for a creator.

        Sweeps the creator's Stash files, adjudicates each against the
        downloaded media (posts + messages), composes one gallery per owning
        item, and flushes in a single batched ``save_all`` (split-scene
        ``stash_id`` is assigned post-flush, then persisted to the metadata DB).

        Args:
            account: The Account being processed
            performer: The Performer for the account
            studio: Optional Studio to associate with galleries
        """
        (
            index,
            item_entities,
            media_with_id,
            split_pairs,
        ) = await self._prepare_file_first(account)
        # Incremental entry point: media already carrying a stash_id are
        # re-verified by-id and dropped from the sweep's index (subsequent runs
        # go straight to the entity instead of rediscovering its file).
        await self._fast_path_known_media(
            index, account, studio, item_entities, media_with_id, split_pairs
        )
        async for file in self._sweep_creator_files():
            await self._safe_adjudicate(
                file, index, account, studio, item_entities, media_with_id, split_pairs
            )
        await self._compose_and_flush(
            account, performer, studio, item_entities, media_with_id, split_pairs
        )

    async def _run_file_first_incremental(
        self,
        account: Account,
        performer: Performer,
        studio: Studio | None,
    ) -> None:
        """Sweep-free incremental file-first pass for the monitoring daemon.

        Builds the media index from just-downloaded content, scans exactly those
        files into Stash (settling before reads), then locates each media's file
        by basename (``find_one`` -> one row; snowflake-unique, override-safe)
        instead of a creator-wide sweep. Reuses the shared adjudication, gallery
        composition, and batched flush. Media whose file is not yet visible to
        Stash are left for a later cycle.

        Args:
            account: The Account being processed
            performer: The Performer for the account
            studio: Optional Studio to associate with galleries
        """
        (
            index,
            item_entities,
            media_with_id,
            split_pairs,
        ) = await self._prepare_file_first(account)
        # Scan exactly the just-downloaded files. local_path is set this run for
        # freshly-downloaded media; absent for prior-cycle media already scanned.
        scan_paths = sorted(
            {
                get_stash_path(Path(media.local_path), self.config)
                for media, _owners in index.values()
                if media.local_path
            }
        )
        if scan_paths:
            await self.scan_creator_folder(paths=scan_paths)
        await self._fast_path_known_media(
            index, account, studio, item_entities, media_with_id, split_pairs
        )
        for _leaf, (media, _owners) in list(index.items()):
            if not media.local_path or not media.local_filename:
                continue  # only this cycle's downloads (local_path is the marker)
            # Match by basename, NOT by path: under override_dldir_w_mapped the
            # Stash library is reorganized, so local paths do not correspond to
            # Stash paths (get_stash_path collapses to the mapped root). The
            # basename carries the snowflake media id, survives the reorg, and is
            # the same key the sweep's leaf-index uses. (Scanning above is by path
            # to make Stash ingest; identity matching is by basename.)
            file = await self.store.find_one(BaseFile, basename=media.local_filename)
            if file is not None:
                await self._safe_adjudicate(
                    file,
                    index,
                    account,
                    studio,
                    item_entities,
                    media_with_id,
                    split_pairs,
                )
        await self._compose_and_flush(
            account, performer, studio, item_entities, media_with_id, split_pairs
        )

    async def _prepare_file_first(
        self, account: Account
    ) -> tuple[
        dict[str, tuple[Media, list[Post | Message]]],
        dict[int, tuple[Post | Message, list[Scene | Image]]],
        list[Media],
        list[tuple[Media, Scene]],
    ]:
        """Build the media index and the empty run-level accumulators.

        A cold preload (STASH_ONLY) resolves belongs_to/habtm but not has_many
        reverse-FK lists, so posts/messages load attachment-less; rebuild those
        lists before the gather filters on them.
        """
        self._reconstruct_attachment_lists()
        self._reconstruct_mention_lists()
        posts = await self._gather_creator_posts(account)
        messages = await self._gather_creator_messages(account)
        index = await self._build_media_index([*posts, *messages])
        # item.id -> (item, [Scene|Image, ...])
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]] = {}
        # Media that got a stash_id this run (owned scene / stamped image).
        media_with_id: list[Media] = []
        # (media, new_scene) — stash_id assigned only AFTER save_all.
        split_pairs: list[tuple[Media, Scene]] = []
        return index, item_entities, media_with_id, split_pairs

    async def _safe_adjudicate(
        self,
        file: BaseFile,
        index: dict[str, tuple[Media, list[Post | Message]]],
        account: Account,
        studio: Studio | None,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Adjudicate one file with per-file isolation — one bad file never aborts."""
        try:
            await self._adjudicate_swept_file(
                file, index, account, studio, item_entities, media_with_id, split_pairs
            )
        except Exception as exc:
            file_id = getattr(file, "id", "?")
            logger.exception(f"Failed to adjudicate file {file_id}", exc_info=exc)
            debug_print(
                {
                    "method": "StashProcessing - run_file_first",
                    "status": "file_adjudication_failed",
                    "file_id": file_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    async def _adjudicate_swept_file(
        self,
        file: BaseFile,
        index: dict[str, tuple[Media, list[Post | Message]]],
        account: Account,
        studio: Studio | None,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Adjudicate one swept file, accumulating its entities in place.

        Processes the file against the media index and, for any entities it
        yields, pairs them with the owning Media/item and records them in the
        run-level accumulators (``item_entities``, ``media_with_id``,
        ``split_pairs``) that the caller flushes once.

        Args:
            file: The swept Stash file to adjudicate
            index: Media index keyed by filename -> (Media, owning items)
            account: The Account being processed
            studio: Optional Studio to associate with entities
            item_entities: Accumulator: item.id -> (item, [Scene|Image, ...])
            media_with_id: Accumulator for Media that got a stash_id this run
            split_pairs: Accumulator for (media, new_scene) split pairs
        """
        if isinstance(file.path, UnsetType):  # sweep fragments always query path
            return
        entities = await self._process_file_first(file, index, account, studio)
        if not entities:
            return
        media, owners = index[PurePath(file.path).name]  # re-lookup to pair
        self._accumulate_entities(
            media, owners, entities, item_entities, media_with_id, split_pairs
        )

    @staticmethod
    def _accumulate_entities(
        media: Media,
        owners: list[Post | Message],
        entities: list[Scene | Image],
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Fan a media's adjudicated entities into every owner's gallery, once.

        The entity joins EVERY owning item's gallery (a shared Media must not
        lose the other items' galleries), but stash_id accumulation is
        once-per-entity, not per-owner.
        """
        for item in owners:
            if item.id is None:  # owners are persisted rows; id is always set
                continue
            item_entities.setdefault(item.id, (item, []))[1].extend(entities)
        for entity in entities:
            if isinstance(entity, Scene) and entity.is_new():
                # A split — owned scenes are not new; stash_id set post-flush.
                split_pairs.append((media, entity))
            else:
                # Owned scene / stamped image already set media.stash_id.
                media_with_id.append(media)

    async def _fast_path_known_media(
        self,
        index: dict[str, tuple[Media, list[Post | Message]]],
        account: Account,
        studio: Studio | None,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Re-verify already-stamped media by-id, before the sweep.

        The incremental entry point of the file-first design: a media that
        already carries a stash_id goes straight to its Scene/Image via
        ``_process_media_fast_path`` (get-by-id + primary re-verify) instead of
        waiting for the sweep to rediscover its file. A leaf is dropped from the
        index only when the fast-path definitively handled it (returned
        entities); a stale/foreign/skip result leaves the leaf for the sweep to
        re-adjudicate, so nothing is silently lost.
        """
        known = [
            (leaf, media, owners)
            for leaf, (media, owners) in list(index.items())
            if media.stash_id is not None
        ]
        for leaf, media, owners in known:
            entities = await self._process_media_fast_path(
                media, owners[0], account, studio
            )
            if not entities:
                continue  # stale/foreign/skip — leave in index for the sweep
            del index[leaf]  # definitively handled; do not re-adjudicate via sweep
            self._accumulate_entities(
                media, owners, entities, item_entities, media_with_id, split_pairs
            )

    async def _compose_and_flush(
        self,
        account: Account,
        performer: Performer,
        studio: Studio | None,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Flush adjudicated entities, then compose one gallery per owning item.

        Ordered so a failure can never leave persistent damage in Stash:

        1. Flush the stamped Scenes/Images and any split scenes FIRST. No gallery
           exists yet, so a failure here cannot orphan one.
        2. Persist ``media.stash_id`` for whatever landed — unconditionally, so a
           later gallery failure does not discard bookkeeping the server already
           accepted.
        3. Compose galleries (per-item isolated) and flush them. Entities now
           carry real IDs, so a new gallery is created with its content attached
           in a single save() rather than created empty and linked later.
        4. Persist the gallery ids onto their posts/messages — same reasoning as
           step 2, and unconditional for the same reason.

        Any phase failure still re-raises, so the creator is reported failed
        rather than as a clean success. The flow is idempotent — entities that
        failed in phase 1 stay dirty and are retried by phase 3's flush, and a
        failed creator self-heals on the next run.
        """
        entity_error = await self._flush_adjudicated_entities(split_pairs)
        await self._persist_media_stash_ids(media_with_id, split_pairs)
        gallery_error = await self._compose_and_flush_galleries(
            account, performer, studio, item_entities
        )
        await self._persist_item_stash_ids(item_entities)
        if entity_error is not None:
            raise entity_error
        if gallery_error is not None:
            raise gallery_error

    async def _flush_adjudicated_entities(
        self,
        split_pairs: list[tuple[Media, Scene]],
    ) -> Exception | None:
        """Flush stamped/split entities and stamp split ids. Returns the error.

        Runs before any gallery is created so a batch failure cannot leave an
        empty gallery behind. Returns the exception instead of raising so the
        caller can still persist the stash_ids that DID land.
        """
        try:
            await self.store.save_all()
        except Exception as exc:
            print_error(f"Failed to flush adjudicated Stash entities: {exc}")
            logger.exception("Failed to flush adjudicated Stash entities", exc_info=exc)
            debug_print(
                {
                    "method": "StashProcessing - run_file_first",
                    "status": "entity_flush_failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return exc
        finally:
            # Split scenes that were actually created now carry a real ID (a
            # partial batch failure leaves the rest new — those keep a temp UUID
            # and must not be int()'d onto the Media).
            for media, scene in split_pairs:
                if not scene.is_new():
                    media.stash_id = int(scene.id)
        return None

    async def _persist_media_stash_ids(
        self,
        media_with_id: list[Media],
        split_pairs: list[tuple[Media, Scene]],
    ) -> None:
        """Persist affected Media rows to the FDNG metadata DB (not Stash).

        Best-effort and per-row isolated: this is bookkeeping for writes Stash
        already accepted, so one bad row must not cost the rest. Dropping a
        stash_id is not fatal (the next run re-adjudicates via the sweep) but it
        forces avoidable rediscovery, so it is worth saving even on a failed run.
        Dedupe by object identity so each Media is saved once.
        """
        fdng_store = get_store()
        affected = media_with_id + [m for m, _ in split_pairs]
        for media in {id(m): m for m in affected}.values():
            if media.stash_id is None:
                continue
            try:
                await fdng_store.save(media)
            except Exception as exc:
                logger.exception(
                    f"Failed to persist stash_id for media {media.id}", exc_info=exc
                )

    async def _persist_item_stash_ids(
        self,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
    ) -> None:
        """Persist each item's gallery id to the FDNG metadata DB.

        ``_get_or_create_gallery`` stamps ``item.stash_id`` in memory so the
        gallery lookup is O(1), but nothing wrote it back — every later run
        (STASH_ONLY especially) re-searched every gallery by code/title/URL and
        risked composing a second one when a title had drifted. Same contract as
        ``_persist_media_stash_ids``: best-effort, per-row isolated, and run even
        after a failed gallery flush so the ids Stash DID accept are kept.
        """
        fdng_store = get_store()
        for item, _entities in item_entities.values():
            if item.stash_id is None:
                continue
            try:
                await fdng_store.save(item)
            except Exception as exc:
                logger.exception(
                    f"Failed to persist stash_id for item {item.id}", exc_info=exc
                )

    async def _compose_and_flush_galleries(
        self,
        account: Account,
        performer: Performer,
        studio: Studio | None,
        item_entities: dict[int, tuple[Post | Message, list[Scene | Image]]],
    ) -> Exception | None:
        """Compose and persist one gallery per owning item. Returns the error.

        Each item's gallery is saved as it is composed, not deferred into one
        creator-wide batch. A pre-existing gallery receives its new content via
        the ``addGalleryImages`` side-mutation, and SGC only fires side mutations
        after every main mutation in the batch has succeeded — so in a single
        batch one transport-level failure (a blip against a remote Stash) drops
        the links for EVERY gallery in the run. Saving per item bounds that loss
        to the one item and lets a long run make durable partial progress.
        """
        first_error: Exception | None = None
        for item, entities in item_entities.values():
            # Per-item isolation: one bad compose/save must not abort the rest.
            try:
                gallery = await self._compose_gallery_for_item(
                    item, entities, account, performer, studio
                )
                if gallery is not None:
                    # No-op for a gallery just created by _get_or_create_gallery
                    # (already saved clean with its content); this is the flush
                    # for an EXISTING gallery that gained links/tags.
                    await self.store.save(gallery)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                print_error(f"Failed to compose gallery for item {item.id}: {exc}")
                logger.exception(
                    f"Failed to compose gallery for item {item.id}", exc_info=exc
                )
                debug_print(
                    {
                        "method": "StashProcessing - run_file_first",
                        "status": "gallery_compose_failed",
                        "item_id": item.id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        try:
            # Sweep-up: chapters, tags created along the way, and any entity left
            # dirty by a failed phase-1 flush (a free retry).
            await self.store.save_all()
        except Exception as exc:
            print_error(f"Failed to flush gallery batch: {exc}")
            logger.exception("Failed to flush gallery batch", exc_info=exc)
            debug_print(
                {
                    "method": "StashProcessing - run_file_first",
                    "status": "gallery_flush_failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if first_error is None:
                first_error = exc
        return first_error

    async def _compose_gallery_for_item(
        self,
        item: Post | Message,
        entities: list[Scene | Image],
        account: Account,
        performer: Performer,
        studio: Studio | None,
    ) -> Gallery | None:
        """Get-or-create the item's gallery with its adjudicated entities linked.

        Linking is done by ``_get_or_create_gallery`` via
        ``_link_gallery_contents`` so that a NEW gallery is created with its
        content already attached (one save()), instead of being persisted empty
        and linked by a later flush that may never land. An EXISTING gallery is
        returned dirty for the caller to save.

        Args:
            item: The owning Post or Message
            entities: Scene/Image entities adjudicated for this item
            account: The Account being processed
            performer: The Performer for the account
            studio: Optional Studio to associate with the gallery

        Returns:
            The item's Gallery, or None when the item has no media content.
        """
        if isinstance(item, Post):
            item_type = "post"
            url_pattern = f"https://fansly.com/post/{item.id}"
        else:
            item_type = "message"
            url_pattern = f"https://fansly.com/messages/{item.groupId}/{item.id}"

        return await self._get_or_create_gallery(
            item=item,
            account=account,
            performer=performer,
            studio=studio,
            item_type=item_type,
            url_pattern=url_pattern,
            entities=entities,
        )
