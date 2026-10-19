"""Account and performer processing mixin."""

from __future__ import annotations

import traceback
import unicodedata
from typing import TYPE_CHECKING

from stash_graphql_client.types import Image, Performer, UnsetType, is_set

from config.schema import (
    FANSLY_USERNAME_ALLOWED_CHARS,
    FANSLY_USERNAME_MAX_LEN,
    FANSLY_USERNAME_MIN_LEN,
)
from metadata import Account, PostMention
from metadata.models import get_store
from textio import print_error, print_warning

from ...logging import debug_print
from ...logging import processing_logger as logger
from ..protocols import StashProcessingProtocol


if TYPE_CHECKING:
    from metadata.entity_store import PostgresEntityStore


def normalize_mention_handle(handle: str | None) -> str | None:
    """Return *handle* as a usable Fansly username, or ``None`` if it is junk.

    Fansly's mention extractor is naive: it emits an ``accountMentions`` entry
    for every ``@`` in a post's text regardless of what follows. A bare ``@``
    before a space — ``"Going live @ 9PM EST"``, ``"See you @ the show"``
    — yields a mention whose handle is the empty string; ``"@10:30PM EST"``
    yields ``"10:30pm"``; a real handle the author followed with punctuation or
    an emoji arrives as ``"example,"`` or ``"sample✨"``; a pasted profile URL
    arrives as ``"/demo"``. Fansly also substitutes the author's styled
    display name for the handle, so a creator shows up as ``"\U0001d5d5\U0001d5fc..."``
    (mathematical-bold ``BoldName``).

    NFKC-fold first, which maps those styled alphabets back to ASCII, then trim
    the characters a Fansly username cannot contain from both ends, then
    keep the result only if what remains is a syntactically valid username. That
    recovers the real handle from the punctuation cases and rejects the rest, so
    a parser artifact never reaches Stash — neither as a blank-valued search
    (which matches the entire library) nor as a performer named after a clock
    time.
    """
    if not handle:
        return None
    handle = unicodedata.normalize("NFKC", handle)
    start, end = 0, len(handle)
    while start < end and handle[start] not in FANSLY_USERNAME_ALLOWED_CHARS:
        start += 1
    while end > start and handle[end - 1] not in FANSLY_USERNAME_ALLOWED_CHARS:
        end -= 1
    cleaned = handle[start:end]
    if not FANSLY_USERNAME_MIN_LEN <= len(cleaned) <= FANSLY_USERNAME_MAX_LEN:
        return None
    if set(cleaned) - FANSLY_USERNAME_ALLOWED_CHARS:
        return None
    return cleaned


def performer_identity(entity: Account | PostMention) -> tuple[str, str] | None:
    """Resolve ``(search_name, username)`` for a performer lookup.

    ``None`` means the entity carries no usable Fansly username, which for a
    ``PostMention`` is Fansly's bare-``@`` artifact (see
    ``normalize_mention_handle``). Callers must skip those rather than look them
    up or create from them.
    """
    if isinstance(entity, PostMention):
        handle = normalize_mention_handle(entity.handle)
        return (handle, handle) if handle else None
    name = entity.displayName or entity.username
    return (name, entity.username) if name and entity.username else None


class AccountProcessingMixin(StashProcessingProtocol):
    """Account and performer processing functionality."""

    async def _find_account(self) -> Account | None:
        """Find account in database using identity map (cache-first).

        Returns:
            Account if found, None otherwise
        """
        store: PostgresEntityStore = get_store()

        if self.state.creator_id is not None:
            # Cache-first: identity map lookup is O(1)
            account = store.get_from_cache(Account, self.state.creator_id)
            if not account:
                account = await store.get(Account, self.state.creator_id)
        else:
            # Search by username (case-insensitive) — filter cache first
            if self.state.creator_name is None:
                return None
            name_lower = self.state.creator_name.lower()
            cached = store.filter(
                Account,
                lambda a: a.username is not None and a.username.lower() == name_lower,
            )
            account = cached[0] if cached else None
            if not account:
                account = await store.find_one(
                    Account, username__iexact=self.state.creator_name
                )

        if not account:
            print_warning(f"No account found for username: {self.state.creator_name}")
        return account

    def _performer_from_account(
        self, account: Account | PostMention, name: str, username: str
    ) -> Performer:
        """Create a Performer object from a Fansly Account (or PostMention).

        This is a local helper that maps Account model fields to the Performer
        type from stash-graphql-client. ``name``/``username`` come from
        ``performer_identity`` so a mention's handle is already normalized and
        validated; passing them in keeps this from minting a performer named
        after one of Fansly's mention-parser artifacts.

        Args:
            account: The Account or PostMention to convert
            name: Display name for the new performer
            username: Fansly username, used as alias and in the profile URL

        Returns:
            Performer object suitable for creating/updating in Stash.
            For new performers, the id field is omitted and auto-generates
            a UUID4 placeholder.
        """
        # Build Fansly profile URL
        url = f"https://fansly.com/{username}"

        # Create Performer without id (auto-generates UUID4 for new objects)
        return Performer(
            name=name,
            alias_list=[username],  # Username as alias for searchability
            urls=[url],
            details=getattr(account, "about", "") or "",  # Biography/about text
        )

    async def _get_or_create_performer(
        self, account: Account | PostMention
    ) -> Performer | None:
        """Get existing performer from Stash or create from account if not found.

        IMPORTANT: Performs deduplication checks (name→alias→URL) BEFORE creating.
        Pattern 1 migration: Use store.find_one() for identity map caching,
        preserving critical sequential deduplication logic.

        The alias and URL steps are substring searches, so they must never run
        with a blank value: Stash reads ``aliases INCLUDES ""`` as ``LIKE '%%'``
        and returns the whole library, whose first row then gets silently
        attached to the scene. ``performer_identity`` rejects the mentions that
        would do that, and the per-step guards below keep any future caller from
        reintroducing it.

        Args:
            account: Account or PostMention to search for or create from

        Returns:
            Performer object from Stash (either found or newly created), or
            ``None`` when the entity carries no usable Fansly username.
        """
        identity = performer_identity(account)
        if identity is None:
            logger.debug(f"Skipping performer lookup for unusable entity: {account!r}")
            return None
        search_name, username = identity
        fansly_url = f"https://fansly.com/{username}"

        # Cache-first: try sync filter() on preloaded performers before GraphQL.
        # Performers are preloaded in _preload_stash_entities() at startup.
        # Sequential name → alias → URL checks prevent duplicates.
        performer = None

        # 1. Exact name match
        results = self.store.filter(Performer, lambda p: p.name == search_name)
        if results:
            logger.debug(f"Cache hit: performer by name: {search_name}")
            performer = results[0]
        if not performer:
            performer = await self.store.find_one(Performer, name__exact=search_name)

        # 2. Alias match (critical for deduplication)
        if not performer and username:
            results = self.store.filter(
                Performer,
                lambda p: (
                    hasattr(p, "alias_list")
                    and is_set(p.alias_list)
                    and username in p.alias_list
                ),
            )
            if results:
                logger.debug(f"Cache hit: performer by alias: {username}")
                performer = results[0]
            if not performer:
                performer = await self.store.find_one(
                    Performer, aliases__contains=username
                )

        # 3. URL match (catches edge cases)
        if not performer and username:
            results = self.store.filter(
                Performer,
                lambda p: (
                    hasattr(p, "urls") and is_set(p.urls) and fansly_url in p.urls
                ),
            )
            if results:
                logger.debug(f"Cache hit: performer by URL: {fansly_url}")
                performer = results[0]
            if not performer:
                performer = await self.store.find_one(
                    Performer, url__contains=fansly_url
                )

        if performer:
            return performer

        # Not found after all deduplication checks - create new performer
        logger.debug(f"Creating new performer for account: {username}")
        performer = self._performer_from_account(account, search_name, username)
        await self.store.save(performer)
        return performer

    async def process_creator(self) -> tuple[Account, Performer]:
        """Process creator metadata into Stash.

        Returns:
            Tuple of (Account, Performer)

        Raises:
            ValueError: If creator_id is not available in state
        """
        try:
            # Find account
            account = await self._find_account()
            debug_print(
                {
                    "method": "StashProcessing - process_creator",
                    "account": account,
                }
            )
            if not account:
                raise ValueError(
                    f"No account found for creator: {self.state.creator_name} "
                    f"(ID: {self.state.creator_id})"
                )

            logger.debug(f"Processing creator: {account.username}")
            # Get or create performer using intelligent fuzzy search
            performer = await self._get_or_create_performer(account)
            if performer is None:  # pragma: no cover — a creator always has a username
                raise ValueError(
                    f"No usable Fansly username for creator: {account.username}"
                )
            logger.debug(f"Obtained performer in Stash: {performer}")
            logger.debug(f"Context client (in process_creator): {self.context}")

            debug_print(
                {
                    "method": "StashProcessing - process_creator",
                    "performer": performer,
                }
            )
            # Handle avatar if needed
            await self._update_performer_avatar(account, performer)
        except Exception as e:
            print_error(f"Failed to process creator: {e}")
            logger.exception("Failed to process creator", exc_info=e)
            debug_print(
                {
                    "method": "StashProcessing - process_creator",
                    "status": "creator_processing_failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        else:
            return account, performer

    async def _update_performer_avatar(
        self, account: Account, performer: Performer
    ) -> None:
        """Update performer's avatar if needed.

        Only updates the avatar if the current image is the default one.

        Args:
            account: Account object containing avatar information
            performer: Performer object to update
        """
        # Account.avatar is a Pydantic relationship — directly accessible
        avatar = account.avatar

        if avatar is None or isinstance(avatar, UnsetType) or not avatar.local_filename:
            debug_print(
                {
                    "method": "StashProcessing - _update_performer_avatar",
                    "status": "no_avatar_found",
                    "account": account.username,
                }
            )
            return

        # Only update if current image is default
        image_path = performer.image_path
        if not is_set(image_path) or image_path is None or "default=true" in image_path:
            # Cache-first: try sync filter on preloaded images, then GraphQL
            filename = avatar.local_filename
            images = self.store.filter(
                Image,
                lambda img: (
                    hasattr(img, "visual_files")
                    and is_set(img.visual_files)
                    and any(
                        is_set(f.path) and filename in f.path for f in img.visual_files
                    )
                ),
            )
            if not images:
                images = await self.store.find(Image, path__contains=filename)
            if not images:
                debug_print(
                    {
                        "method": "StashProcessing - _update_performer_avatar",
                        "status": "no_avatar_found",
                        "account": account.username,
                    }
                )
                return
            # Use first image (sorted by created_at in Stash)
            avatar_img = images[0]
            visual_files = avatar_img.visual_files
            if isinstance(visual_files, UnsetType) or not visual_files:
                return
            avatar_path = visual_files[0].path
            if isinstance(avatar_path, UnsetType):
                return
            try:
                await performer.update_avatar(self.context.client, avatar_path)
                debug_print(
                    {
                        "method": "StashProcessing - _update_performer_avatar",
                        "status": "avatar_updated",
                        "performer": performer.name,
                    }
                )
            except Exception as e:
                print_error(f"Failed to update performer avatar: {e}")
                logger.exception("Failed to update performer avatar", exc_info=e)
                debug_print(
                    {
                        "method": "StashProcessing - _update_performer_avatar",
                        "status": "avatar_update_failed",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )

    async def _find_existing_performer(
        self, account: Account | PostMention
    ) -> Performer | None:
        """Find existing performer in Stash using identity map.

        Args:
            account: Account to find performer for

        Returns:
            Performer data if found, None otherwise
        """
        # Cache-first: try sync get_cached() (zero-cost after preload),
        # fall back to async get() only if cache misses.
        # Uses getattr for duck-typing: works with both Account (stash_id/username)
        # and PostMention (handle) objects after the Pydantic migration.
        stash_id = getattr(account, "stash_id", None)
        if stash_id:
            try:
                performer = self.store.get_cached(Performer, str(stash_id))
                if not performer:
                    performer = await self.store.get(Performer, str(stash_id))
                if performer:
                    debug_print(
                        {
                            "method": "StashProcessing - _find_existing_performer",
                            "stash_id": stash_id,
                            "performer": performer,
                            "cached": "identity_map",
                        }
                    )
                    return performer
            except Exception as e:
                logger.debug(f"Failed to get performer by stash_id: {e}")

        # Fallback to name search (cache-first, then GraphQL)
        # Supports both Account (username) and PostMention (handle) via duck-typing
        name = account.username if isinstance(account, Account) else account.handle
        if not name:  # pragma: no cover — username and handle are required str fields
            return None
        results = self.store.filter(Performer, lambda p: p.name == name)
        performer = results[0] if results else None
        if not performer:
            performer = await self.store.find_one(Performer, name=name)
        if performer:
            debug_print(
                {
                    "method": "StashProcessing - _find_existing_performer",
                    "username": name,
                    "performer": performer,
                }
            )
        return performer

    async def _update_account_stash_id(
        self,
        account: Account,
        performer: Performer,
    ) -> None:
        """Update account's stash ID and persist to database.

        Supports both Account (has stash_id field) and PostMention (skipped).

        Args:
            account: Account (or PostMention) to update
            performer: Performer containing the stash ID
        """
        if not hasattr(account, "stash_id"):
            return  # PostMention doesn't have stash_id
        store: PostgresEntityStore = get_store()
        account.stash_id = int(performer.id)
        await store.save(account)
