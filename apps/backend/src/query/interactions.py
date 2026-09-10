"""Reconstruct customer/brand Interactions from the raw tweet dataset.

Turns the flat twcs tweet table into Interactions in the sense of CONTEXT.md:
each Interaction is an ordered chain of customer and brand turns, rooted at the
opening Customer Message and attributed to a single Brand (SpotifyCares in v0).

The dataset records two pointers per tweet:

* ``in_response_to_tweet_id`` - the tweet this one replies to (single parent)
* ``response_tweet_id`` - comma-separated ids of tweets that replied to this
  one; it can include indirect mentions and is not needed for reconstruction

Stitching therefore follows ``in_response_to_tweet_id`` only, and orders turns
by ``created_at`` (tweet ids are not monotonic with reply time in this dataset).

The pass over the CSV works in stages:

1. read every row, validate it, and index it by tweet id (ids are unique)
2. collect seeds: inbound tweets that engage the Brand (mention it, are replied
   to by it, or reply to it)
3. climb from each seed to its opening customer message: through intervening
   brand turns and through same-author ancestors that also engage the Brand
4. grow each opening into a dyad: tweets by the customer author or the Brand
5. drop openings that lie inside another opening's dyad (continuations, not
   new Interactions) so every turn belongs to exactly one Interaction
6. emit an Interaction when the dyad holds at least one brand turn

An Interaction must contain at least one brand turn: without a brand reply no
closure verdict (Resolved / Uncertain / Unresolved) can ever be produced, so the
chain is reported as an unanswered opening instead.
"""

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TWCS_PATH = REPO_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
DEFAULT_INTERACTIONS_PATH = REPO_ROOT / "data" / "interactions.jsonl"
DEFAULT_BRAND = "SpotifyCares"
COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)
TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"


class InteractionsError(Exception):
    """Raised when the source CSV cannot be turned into Interactions."""


@dataclass(frozen=True)
class Turn:
    """A single customer or brand message inside an Interaction."""

    tweet_id: int
    author_id: str
    side: Literal["customer", "brand"]
    created_at: datetime
    text: str


@dataclass(frozen=True)
class Interaction:
    """A normalized Interaction: one customer, one brand, ordered turns."""

    interaction_id: int  # tweet id of the opening Customer Message
    customer_id: str
    brand_id: str
    turns: tuple[Turn, ...]

    @property
    def customer_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.side == "customer")

    @property
    def brand_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.side == "brand")

    @property
    def opening_message(self) -> Turn:
        return self.turns[0]


@dataclass(frozen=True)
class InteractionsReport:
    """Counts for every stitching stage, for honest reporting."""

    brand_id: str
    total_rows: int
    inbound_rows: int
    brand_rows: int
    seed_count: int
    opening_count: int
    absorbed_openings: int
    interactions: int
    unanswered_openings: int
    turns_total: int
    turns_customer: int
    turns_brand: int


@dataclass(frozen=True)
class _Row:
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    parent_id: int | None


def build_interactions(
    csv_path: Path | str = DEFAULT_TWCS_PATH,
    brand_id: str = DEFAULT_BRAND,
) -> tuple[tuple[Interaction, ...], InteractionsReport]:
    """Stitch the raw tweets into Interactions for one brand.

    Returns the Interactions (ordered by opening tweet id) and a report with
    counts for every stitching stage.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise InteractionsError(f"source CSV does not exist: {csv_path}")

    rows: dict[int, _Row] = {}
    children: dict[int, list[int]] = {}
    total_rows = 0
    inbound_rows = 0
    brand_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        _require_header(next(reader, None), csv_path)
        for record in reader:
            if not record:
                continue
            tweet = _parse_row(record, csv_path)
            total_rows += 1
            if tweet.tweet_id in rows:
                raise InteractionsError(
                    f"duplicate tweet_id {tweet.tweet_id} in {csv_path}"
                )
            rows[tweet.tweet_id] = tweet
            if tweet.inbound:
                inbound_rows += 1
            if tweet.author_id == brand_id:
                brand_rows += 1
            if tweet.parent_id is not None:
                children.setdefault(tweet.parent_id, []).append(tweet.tweet_id)

    brand_replied_to: set[int] = set()
    for tweet in rows.values():
        if tweet.author_id != brand_id or tweet.parent_id is None:
            continue
        if tweet.parent_id in rows:
            brand_replied_to.add(tweet.parent_id)

    seeds = _find_seeds(rows, brand_replied_to, brand_id)

    opening_of_seed: dict[int, int] = {}
    for seed in seeds:
        opening_of_seed[seed] = _climb_to_opening(seed, rows, brand_replied_to, brand_id)

    dyad_of_opening: dict[int, set[int]] = {}
    for opening in set(opening_of_seed.values()):
        dyad_of_opening[opening] = _grow_dyad(
            opening, rows, children, rows[opening].author_id, brand_id
        )

    # An opening that lies inside another opening's dyad is a continuation of
    # that Interaction, not a new one (its dyad is a subset of the container's).
    # Dropping it keeps the emitted Interactions pairwise disjoint, so no turn
    # is ever counted twice.
    absorbed_openings = set()
    for opening, dyad in dyad_of_opening.items():
        for other in dyad:
            if other != opening and other in dyad_of_opening:
                absorbed_openings.add(other)

    interactions: list[Interaction] = []
    unanswered_openings = 0
    turns_total = 0
    turns_customer = 0
    turns_brand = 0
    for opening in sorted(dyad_of_opening):
        if opening in absorbed_openings:
            continue
        dyad = dyad_of_opening[opening]
        brand_turn_ids = [t for t in dyad if rows[t].author_id == brand_id]
        if not brand_turn_ids:
            unanswered_openings += 1
            continue
        ordered = sorted(dyad, key=lambda t: (rows[t].created_at, t))
        turns = tuple(
            Turn(
                tweet_id=t,
                author_id=rows[t].author_id,
                side="brand" if rows[t].author_id == brand_id else "customer",
                created_at=rows[t].created_at,
                text=rows[t].text,
            )
            for t in ordered
        )
        turns_total += len(turns)
        turns_customer += sum(1 for t in dyad if rows[t].author_id != brand_id)
        turns_brand += len(brand_turn_ids)
        interactions.append(
            Interaction(
                interaction_id=opening,
                customer_id=rows[opening].author_id,
                brand_id=brand_id,
                turns=turns,
            )
        )

    report = InteractionsReport(
        brand_id=brand_id,
        total_rows=total_rows,
        inbound_rows=inbound_rows,
        brand_rows=brand_rows,
        seed_count=len(seeds),
        opening_count=len(dyad_of_opening),
        absorbed_openings=len(absorbed_openings),
        interactions=len(interactions),
        unanswered_openings=unanswered_openings,
        turns_total=turns_total,
        turns_customer=turns_customer,
        turns_brand=turns_brand,
    )
    return tuple(interactions), report


def interaction_to_json(interaction: Interaction) -> dict:
    """Serialize an Interaction to a JSON-compatible mapping."""
    return {
        "interaction_id": interaction.interaction_id,
        "customer_id": interaction.customer_id,
        "brand_id": interaction.brand_id,
        "turns": [
            {
                "tweet_id": turn.tweet_id,
                "author_id": turn.author_id,
                "side": turn.side,
                "created_at": turn.created_at.isoformat(),
                "text": turn.text,
            }
            for turn in interaction.turns
        ],
    }


def write_interactions_jsonl(
    interactions: tuple[Interaction, ...], output_path: Path | str
) -> Path:
    """Write Interactions as JSON Lines, swapping the file in atomically.

    Each line is one Interaction with its ordered turns, in the normalized
    format downstream stages (English filter, sampling) consume.
    """
    output_path = Path(output_path)
    temporary_path = stage_interactions_jsonl(interactions, output_path)
    try:
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return output_path


def stage_interactions_jsonl(
    interactions: tuple[Interaction, ...], output_path: Path | str
) -> Path:
    """Write Interactions to a temporary sibling, ready to be swapped in.

    Callers that must update several files as one transaction stage every
    output first and then swap the returned paths in together, so a write
    failure cannot leave a partial set of new files behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for interaction in interactions:
                handle.write(json.dumps(interaction_to_json(interaction)) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def read_interactions_jsonl(
    input_path: Path | str = DEFAULT_INTERACTIONS_PATH,
) -> tuple[Interaction, ...]:
    """Read Interactions back from the JSON Lines format written above.

    Inverse of :func:`write_interactions_jsonl`: parses every record and
    validates its JSON Lines structure and unique interaction ids so downstream
    stages never see a structurally malformed Interaction. Returns the
    Interactions in file order.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise InteractionsError(f"interactions JSONL does not exist: {input_path}")
    interactions = []
    seen_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"line {line_number} of {input_path}"
            interaction = _parse_interaction_line(line, location)
            if interaction.interaction_id in seen_ids:
                raise InteractionsError(
                    f"duplicate interaction_id {interaction.interaction_id} on {location}"
                )
            seen_ids.add(interaction.interaction_id)
            interactions.append(interaction)
    return tuple(interactions)


def _parse_interaction_line(line: str, location: str) -> Interaction:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise InteractionsError(f"malformed JSON on {location}: {exc}") from exc
    if not isinstance(record, dict):
        raise InteractionsError(f"malformed Interaction on {location}: expected an object")
    interaction_id = record.get("interaction_id")
    customer_id = record.get("customer_id")
    brand_id = record.get("brand_id")
    turns_raw = record.get("turns")
    if type(interaction_id) is not int:
        raise InteractionsError(f"malformed Interaction on {location}: invalid interaction_id")
    if not isinstance(customer_id, str) or not isinstance(brand_id, str):
        raise InteractionsError(f"malformed Interaction on {location}: invalid author ids")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise InteractionsError(f"malformed Interaction on {location}: turns must be non-empty")
    turns = tuple(
        _parse_turn_line(turn, location, index)
        for index, turn in enumerate(turns_raw)
    )
    if turns[0].side != "customer":
        raise InteractionsError(
            f"malformed Interaction on {location}: first turn must be a customer message"
        )
    return Interaction(
        interaction_id=interaction_id,
        customer_id=customer_id,
        brand_id=brand_id,
        turns=turns,
    )


def _parse_turn_line(turn: object, location: str, index: int) -> Turn:
    location = f"{location}, turn {index}"
    if not isinstance(turn, dict):
        raise InteractionsError(f"malformed Turn on {location}: expected an object")
    tweet_id = turn.get("tweet_id")
    author_id = turn.get("author_id")
    side = turn.get("side")
    created_at = turn.get("created_at")
    text = turn.get("text")
    if type(tweet_id) is not int:
        raise InteractionsError(f"malformed Turn on {location}: invalid tweet_id")
    if not isinstance(author_id, str) or side not in ("customer", "brand"):
        raise InteractionsError(f"malformed Turn on {location}: invalid author_id or side")
    if not isinstance(text, str):
        raise InteractionsError(f"malformed Turn on {location}: invalid text")
    if not isinstance(created_at, str):
        raise InteractionsError(f"malformed Turn on {location}: invalid created_at")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise InteractionsError(f"malformed Turn on {location}: invalid created_at") from exc
    return Turn(
        tweet_id=tweet_id,
        author_id=author_id,
        side=side,
        created_at=timestamp,
        text=text,
    )


def _parse_row(record: list[str], csv_path: Path) -> _Row:
    try:
        tweet_id = int(record[0])
        author_id = record[1]
        inbound = _parse_inbound(record[2])
        created_at = _parse_timestamp(record[3])
        text = record[4]
        parent_raw = record[6].strip()
        parent_id = int(parent_raw) if parent_raw else None
    except (IndexError, ValueError) as exc:
        raise InteractionsError(f"malformed row in {csv_path}: {record!r}") from exc
    return _Row(tweet_id, author_id, inbound, created_at, text, parent_id)


def _parse_inbound(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid inbound value: {value!r}")


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _require_header(header: list[str] | None, csv_path: Path) -> None:
    if header is None:
        raise InteractionsError(f"CSV file has no header: {csv_path}")
    if tuple(header) != COLUMNS:
        raise InteractionsError(
            f"unexpected columns in {csv_path}: expected {COLUMNS}, found {tuple(header)}"
        )


def _find_seeds(
    rows: dict[int, _Row], brand_replied_to: set[int], brand_id: str
) -> set[int]:
    """Inbound tweets that engage the Brand: mention it, or are adjacent to it.

    A seed engages the Brand when its text mentions the Brand handle, when the
    Brand replied to it, or when it replies to a Brand tweet. Mention matching
    requires a whole handle (``@brand`` anywhere in the text, ending at a
    handle boundary) so a handle like ``@SpotifyCaresHelp`` is not matched by
    ``@SpotifyCares``.
    """
    seeds: set[int] = set()
    for tweet in rows.values():
        if _is_seed(tweet, rows, brand_replied_to, brand_id):
            seeds.add(tweet.tweet_id)
    return seeds


def _mentions_brand(text: str, mention: str) -> bool:
    """True when the tweet text mentions the Brand handle as a whole handle.

    The mention may appear anywhere in the text (replies put it first, but
    fresh mentions often trail the message), and must end at a handle
    boundary, so ``@SpotifyCaresHelp`` is NOT matched by ``@SpotifyCares``.
    """
    lowered = text.lower()
    start = 0
    while True:
        index = lowered.find(mention, start)
        if index == -1:
            return False
        rest = lowered[index + len(mention):]
        if not rest or not (rest[0].isalnum() or rest[0] == "_"):
            return True
        start = index + 1


def _climb_to_opening(
    seed: int,
    rows: dict[int, _Row],
    brand_replied_to: set[int],
    brand_id: str,
) -> int:
    """Walk up from a seed to the opening customer message.

    The climb passes through intervening brand turns (a customer answering a
    brand question is a continuation, not a new Interaction) and through
    same-author ancestors that themselves engage the Brand (customer
    self-reply chains). The opening is the highest customer-authored message
    reached; brand turns are climbed through but never become the opening.
    """
    author = rows[seed].author_id
    current = seed
    opening = seed
    visited: set[int] = {seed}
    while True:
        parent_id = rows[current].parent_id
        if parent_id is None or parent_id == current or parent_id in visited:
            return opening
        parent = rows.get(parent_id)
        if parent is None:
            return opening
        if parent.author_id == brand_id:
            visited.add(parent_id)
            current = parent_id
            continue
        if (
            parent.author_id == author
            and _is_seed(parent, rows, brand_replied_to, brand_id)
        ):
            visited.add(parent_id)
            current = parent_id
            opening = parent_id
            continue
        return opening


def _is_seed(
    tweet: _Row,
    rows: dict[int, _Row],
    brand_replied_to: set[int],
    brand_id: str,
) -> bool:
    """Seed predicate reusable during climbing (same rule as _find_seeds)."""
    if not tweet.inbound or tweet.author_id == brand_id:
        return False
    mention = "@" + brand_id.lower()
    if _mentions_brand(tweet.text, mention):
        return True
    if tweet.tweet_id in brand_replied_to:
        return True
    parent = rows.get(tweet.parent_id) if tweet.parent_id is not None else None
    return parent is not None and parent.author_id == brand_id


def _grow_dyad(
    opening: int,
    rows: dict[int, _Row],
    children: dict[int, list[int]],
    customer_id: str,
    brand_id: str,
) -> set[int]:
    """Collect the reachable subgraph authored by the customer or the Brand."""
    dyad = {opening}
    stack = [opening]
    while stack:
        current = stack.pop()
        for child_id in children.get(current, ()):
            if child_id in dyad:
                continue
            child_author = rows[child_id].author_id
            if child_author == customer_id or child_author == brand_id:
                dyad.add(child_id)
                stack.append(child_id)
    return dyad
