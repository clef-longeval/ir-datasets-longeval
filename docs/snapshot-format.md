# LongEval Snapshot Format (draft, `format_version` 2)

A file layout for dynamic test collections that are created as a series of snapshots over time. One loader in `ir_datasets_longeval` reads any collection in this format and exposes it through two timelines: `time` (snapshots by calendar date) and `age` (queries by time since their first appearance).

The format is meant to fit existing producers with minimal changes. Reference examples: `tests/resources/dynamic-dataset` (daily news collection) and `tests/resources/minimal-release` (LongEval-Sci). See [Producer changes](#producer-changes) for what each one needs to change.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## Collection layout

A collection is a root folder that contains one subfolder per snapshot:

```
<collection-root>/
├── <snapshot>/
│   ├── manifest.json            required
│   ├── documents/               required
│   │   ├── documents_000000.jsonl.gz
│   │   └── …
│   ├── queries.jsonl.gz         required: queries.jsonl[.gz] or queries.tsv[.gz]
│   ├── topics.jsonl.gz          optional
│   ├── qrels.trec.gz            optional: default qrels set
│   ├── qrels-<name>.trec.gz     optional: named qrels sets
│   ├── run.trec.gz              optional: default run
│   ├── run-<name>.trec.gz       optional: named runs
│   └── …                        anything else is ignored (e.g. serps/)
└── <snapshot>/
    └── …
```

- The snapshot folder name has no meaning. Snapshots are identified by `snapshot` and ordered by `from` in their manifest.
- Every data file MAY be gzip-compressed and then carries a `.gz` suffix. The loader decides from the suffix. A snapshot MUST NOT contain both the compressed and the uncompressed version of the same file.
- Files and folders not named in this spec are ignored, so producers can ship extra material alongside.
- There are no built-in topic types or qrels types. Collections that differ in kind (e.g. the news pipeline's `dynamic` and `static` topics) are released as separate collection roots.

## `manifest.json`

One per snapshot. Unknown keys are allowed and ignored, so producers can keep their own provenance fields (e.g. Sci's `selection`, `relevance`, `counts`).

| Key | Required | Type | Meaning |
|---|---|---|---|
| `format` | yes | string | Always `"longeval-snapshot"`. |
| `format_version` | yes | integer | `2` for this spec. |
| `snapshot` | yes | string | Snapshot ID, unique within the collection. Used in dataset IDs, so it MUST NOT contain `/`. |
| `from` | yes | ISO 8601 date or datetime | Start of the time span the snapshot covers (inclusive). |
| `to` | yes | ISO 8601 date or datetime | End of the time span (exclusive). |
| `period` | yes | ISO 8601 duration | Snapshot length, built from years, months, weeks and days only (`P1D`, `P1W`, `P3M`, …). MUST be the same in every snapshot of a collection. |
| `collection` | no | string | Collection name, used as the dataset ID prefix. Defaults to the root folder name. If set, it MUST be the same in every snapshot. |
| `documents.id_field` | no | string | Document field that holds the document ID. Default `"doc_id"`. |
| `documents.text_fields` | no | list of strings | Fields joined (newline-separated, skipping missing or null values) to form the document text. Default `["title", "text"]`. |
| `documents.fields` | no | list of strings | Every document field occurring in the collection. If omitted, the loader discovers them (see [Fields](#fields)). |
| `documents.types`, `queries.types`, `topics.types` | no | object | Field types to parse, keyed by field name: `"date"` or `"datetime"` (ISO 8601). See [Fields](#fields). |
| `qrels` | no | object | Human-readable descriptions of qrels sets, keyed by set name; `""` for the default set. |
| `runs` | no | object | Human-readable descriptions of runs, keyed by run name; `""` for the default run. |
| `files` | no | object | sha256 checksums of data files, keyed by path relative to the snapshot. The loader MAY verify them. |

`documents`, `queries` and `topics` are objects; other keys in them (e.g. Sci's `path`, `shards`, `selection`) are ignored.

### Time spans

`from`/`to` are end-exclusive: a daily snapshot for 6 November 2026 is `"from": "2026-11-06", "to": "2026-11-07"`.

All `from`/`to` values of a collection MUST be of the same kind: either all dates (`2026-11-06`) or all datetimes. Datetimes MUST carry a UTC offset (`2024-11-04T00:00:00+00:00`), and the loader compares and steps them in UTC.

All snapshots of a collection lie on one grid. With `t₀` the earliest `from` in the collection, every snapshot MUST satisfy:

- `to == from + period`
- `from == t₀ + n × period` for some integer `n ≥ 0` (its *slot*)

Durations are added in calendar arithmetic, so `P3M` steps from 1 March to 1 June to 1 September even though those spans differ in days. Slots with no snapshot are **holes**. They are legal and represent missing data. Producers MUST NOT fill them with copies of neighbouring snapshots.

## Fields

Documents, queries and topics carry arbitrary fields, and every field is exposed as an attribute (`doc.author`, `doc.publish_date`, `query.llm_model`, `query.topic.narrative`).

- **Which fields:** for each kind (documents, queries, topics), every field name that occurs in any record of the collection, across all snapshots. Time and age views therefore share one record type per kind. A record without a field has `None` for it.
- **Discovery:** document fields MAY be declared in `documents.fields`. If declared, the list MUST be complete. If omitted, the loader scans the documents to find them. Query and topic fields are always discovered (the files are small).
- **Values** are exposed as parsed from JSON (string, number, boolean, list, object, `null`). Fields listed in `<kind>.types` as `"date"` or `"datetime"` are parsed from ISO 8601. A value that does not parse becomes `None`, with a warning.
- **Attribute names** equal field names, except:
  - characters other than letters, digits and `_` become `_`, and a name starting with a digit or `_` gets the prefix `f` (`publish-date` → `publish_date`, `_score` → `f_score`);
  - Python keywords, `count`, `index`, and the reserved names below get a trailing `_` (`class` → `class_`).

  Two fields that end up with the same name are an error.
- **Reserved names** are set by the loader:

  | Kind | Reserved | Meaning |
  |---|---|---|
  | documents | `doc_id` | the document ID, taken from `documents.id_field` |
  | documents | `snapshot` | age view only: the snapshot the version comes from |
  | queries | `query_id`, `query` | ID and text |
  | queries | `topic` | the joined topic record, or `None` |
  | queries | `snapshot` | age view only: the query's age-`k` snapshot |
  | topics | `topic_id` | the topic ID |

  The ID field itself is exposed only as `doc_id`, not under its original name. If `documents.id_field` is not `doc_id`, the documents MUST NOT also contain a field named `doc_id`.

## Documents

`documents/` contains one or more JSONL files (`*.jsonl` or `*.jsonl.gz`), read in name order. Other files in the folder (e.g. a shard `manifest.json`) are ignored.

- Each line is one JSON object with arbitrary fields, exposed as described in [Fields](#fields).
- The field named by `documents.id_field` MUST be present, a non-empty string, and unique within the snapshot.
- The text fields from `documents.text_fields` MAY be missing or `null`. `default_text()` joins them; they also remain ordinary attributes.

**Document IDs across snapshots.** Document IDs only need to be unique within a snapshot. The same ID MAY appear in several snapshots, with the same or different content; each occurrence is a separate version of that document. The time view uses the IDs as released. The age view, which combines snapshots, makes versions distinct by prefixing the snapshot (see [Age](#age)), so producers do not need to version IDs themselves.

## Queries

Exactly one of:

- `queries.tsv[.gz]`: tab-separated, header row `query_id<TAB>query`, one query per line.
- `queries.jsonl[.gz]`: one JSON object per line with required `query_id` and `query`. All other fields become attributes (see [Fields](#fields)). `topic_id` links to a topic (see below).

Query IDs MUST be unique within a snapshot and MUST stay stable across snapshots: the same information need keeps the same `query_id` in every snapshot it appears in. The query text MUST NOT change between snapshots: a different text needs a new `query_id`. The age timeline depends on both rules.

## Topics (optional)

`topics.jsonl[.gz]`, one JSON object per line with required `topic_id`. All other fields (e.g. `title`, `description`, `narrative`) become attributes (see [Fields](#fields)). Queries carrying a `topic_id` are joined to the topic of the same snapshot and expose it as `query.topic`. Every `topic_id` used by a query in a snapshot MUST exist in that snapshot's topics file.

Topics are descriptive only. The loader does not use topic fields such as `valid_from` to place queries on a timeline.

## Qrels (optional)

TREC qrels: four whitespace-separated columns `query_id iteration doc_id relevance`, with `relevance` an integer.

| File name | Qrels set |
|---|---|
| `qrels.trec[.gz]` | default set |
| `qrels-<name>.trec[.gz]` | named set `<name>`; `<name>` matches `[a-z0-9]+(-[a-z0-9]+)*` |

- Every `query_id` MUST occur in the same snapshot's queries.
- Every `doc_id` SHOULD occur in the same snapshot's documents.
- A snapshot MAY lack any set, including the default one. That is a hole for that set, not an error.
- The set names in a collection are the union of all names found across its snapshots.

## Runs (optional)

TREC runs: six whitespace-separated columns `query_id Q0 doc_id rank score tag`, e.g. baselines or scraped search engine rankings. Named like qrels sets:

| File name | Run |
|---|---|
| `run.trec[.gz]` | default run |
| `run-<name>.trec[.gz]` | named run `<name>`; same name rule as qrels sets |

- Every `query_id` MUST occur in the same snapshot's queries.
- Every `doc_id` SHOULD occur in the same snapshot's documents.
- A snapshot MAY lack any run. That is a hole for that run, not an error.
- The run names in a collection are the union of all names found across its snapshots.

Runs are not part of dataset IDs (combined with qrels sets, the number of IDs would multiply). Every dataset instead exposes:

- `scoreddocs_iter()`: the default run, as ir_datasets scored documents; empty if there is none
- `get_runs()`: the run names of the collection, `""` for the default run
- `runs_iter(name)`: the scored documents of run `name`

## Timelines and dataset IDs

The loader registers every collection under two timelines. `<set>` is a named qrels set. IDs without `/<set>` use the default set and have no qrels if the collection has none.

| ID | Content |
|---|---|
| `<collection>/time/<snapshot>[/<set>]` | One snapshot as released. |
| `<collection>/time[/<set>]` | Meta dataset of all `time/…` datasets, in time order. |
| `<collection>/age/<k>[/<set>]` | Every query at age `k`, i.e. `k` periods after its first appearance. |
| `<collection>/age[/<set>]` | Meta dataset of all `age/…` datasets, for `k = 0, 1, …`. |

`load(path)` on a collection root returns the same structure for a local folder.

### Time

One dataset per snapshot:

- Documents, queries, topics, qrels and runs come from that snapshot only.
- `get_snapshot()` returns the snapshot ID. `get_timestamp()` returns `from`.
- `get_prior_datasets()` returns the earlier snapshots, newest first.
- Holes do not appear as datasets. The loader reports them as warnings.

### Age

Aligns queries by their age rather than by calendar date.

- **First appearance** of a query is the `from` of the earliest snapshot that contains its `query_id`.
- **Age `k`** of a query is the snapshot whose `from` equals `first appearance + k × period`.
- Dataset `age/<k>` contains:
  - **queries:** each query whose age-`k` snapshot exists and contains it, with the field values from that snapshot and the attribute `snapshot` holding that snapshot's ID;
  - **qrels and runs:** for those queries, taken from their age-`k` snapshots, with document IDs rewritten to `<snapshot>/<doc_id>`;
  - **documents:** every version of every document from all snapshots, with IDs `<snapshot>/<doc_id>` and the attribute `snapshot`. Snapshot IDs never contain `/`, so the first `/` separates the two parts.
- If a query's age-`k` slot is a hole, or the snapshot no longer contains the query, the query is absent from `age/<k>` (and may reappear at a later `k`).
- `k` runs from 0 up to the largest age at which any query still appears.
- `get_snapshot()` returns `k` and `get_timestamp()` returns `None`. `get_prior_datasets()` returns `age/<k-1>` … `age/0`.

All `age/<k>` datasets share one document store for the collection, built once. Qrels and runs always point to the document version from the query's own snapshot.

**Choosing the search space.** Searching all documents works with any tool, but a query then also sees other versions of the same document and documents from later snapshots. Those are unjudged and count as non-relevant. To reproduce what a system could have found at the time, restrict each query to the documents with prefix `query.snapshot + "/"`, e.g. by indexing the snapshot as a field and filtering at query time. Both setups use the same document IDs and the same qrels.

## Requirements

The loader assumes a valid collection and does not check it; a collection that breaks these rules may fail to load or load incorrectly. A collection is invalid when:

- a snapshot has no `manifest.json`, or a required manifest key is missing or malformed
- `format` is not `"longeval-snapshot"` or `format_version` is unsupported
- `period` or `collection` differ between snapshots, or two snapshots share a `snapshot` ID or a slot
- `from`/`to` mix dates and datetimes within the collection, or a datetime has no UTC offset
- a snapshot is off the grid (`to != from + period`, or `from` is not on a slot)
- `documents/` or the queries file is missing, or both compressed and uncompressed versions of a file exist
- document or query IDs are duplicated within a snapshot
- a `query_id` has different query text in different snapshots
- two fields map to the same attribute name, a document has a field `doc_id` while `documents.id_field` names another field, or a document has a field missing from a declared `documents.fields`
- a qrels or run `query_id` is not in the snapshot's queries, or a query's `topic_id` is not in its topics

It is valid but likely incomplete when:

- there are holes in the grid
- a qrels or run `doc_id` is not in the snapshot's documents
- a checksum in `files` does not match
- a value of a field declared in `<kind>.types` does not parse

## Producer changes

### News pipeline (`conf26-temporal-topics` → `dynamic-dataset`)

- Write one collection root per topic type (`dynamic`, `static`); one folder per day as today.
- Add `manifest.json` per day:
  ```json
  {
    "format": "longeval-snapshot",
    "format_version": 2,
    "snapshot": "2026-11-06",
    "from": "2026-11-06",
    "to": "2026-11-07",
    "period": "P1D",
    "documents": {"types": {"fetched_at": "datetime", "publish_date": "date"}},
    "qrels": {"in": "…", "re": "…"}
  }
  ```
  `id_field` and `text_fields` keep their defaults (`doc_id`; `title`, `text`).
- Move `documents.jsonl.gz` to `documents/documents_000000.jsonl.gz`.
- Use the base `doc_id` (unique within a day) as document ID in qrels and runs instead of the dated `docno` (`<id>.<date>`). The age view's `<snapshot>/` prefix takes over versioning. The `docno` field can be dropped from documents.
- Rename `qrels.in.trec.gz` / `qrels.re.trec.gz` to `qrels-in.trec.gz` / `qrels-re.trec.gz` (the example's `qrels.txt` to `qrels.trec.gz` or the matching named set).
- Rename `run.tsv.gz` to `run.trec.gz`.
- No change needed to queries, topics or `serps/`.

### LongEval-Sci (`minimal-release`)

- In `manifest.json`, set `format_version` to `2` and add top-level `from`, `to` and `period` (currently only nested under `selection.snapshot`), plus document field settings:
  ```json
  {
    "format": "longeval-snapshot",
    "format_version": 2,
    "snapshot": "snapshot-1",
    "from": "2024-11-04T00:00:00+00:00",
    "to": "2024-11-06T00:00:00+00:00",
    "period": "P2D",
    "documents": {"id_field": "id", "text_fields": ["title", "abstract", "fullText"], "…": "existing keys"}
  }
  ```
- No change needed to documents, queries or qrels.

## Future Work

- **`period` granularity:** sub-day periods (`PT1H`) are excluded for now. Supporting them would mean:
  - allowing clock units (H/M/S) in `period`, but not mixed with calendar units (no `P1DT6H`);
  - requiring datetimes for `from`/`to` whenever `period` uses clock units;
  - registering only `age/<k>` datasets that contain queries, since hourly periods produce thousands of ages.
- **Age view storage:** the shared document store keeps every version in full, even when a document is unchanged between snapshots. Fine for news (~100k documents); for large collections such as Sci, where each snapshot repeats a mostly unchanged corpus, the store could keep identical content once and answer for every versioned ID. This is a loader optimisation and does not affect the format.