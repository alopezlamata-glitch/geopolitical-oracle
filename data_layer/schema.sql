-- =============================================================================
-- Geopolitical Oracle — Lakehouse Temporal Schema
-- =============================================================================
-- Engine: DuckDB (primary), exportable to Parquet for batch training.
--
-- Design rules:
--   1. Nothing is ever overwritten. All mutations create new versions.
--   2. Every row that touches time has `valid_from` / `valid_to` or
--      `as_of_time`. This makes "as of T" reconstruction exact.
--   3. `data_lineage` tracks every transformation: who produced what from what.
--   4. `feature_snapshots` captures the exact feature vector seen by the model
--      at prediction time — not recomputed later.
--
-- Partition strategy (for Parquet export):
--   raw_documents       → source / published_date
--   canonical_documents → source / published_date
--   canonical_events    → event_type / event_date
--   entity_relations    → entity_country / valid_from
--   feature_snapshots   → question_predicate / as_of_date
-- =============================================================================


-- =============================================================================
-- LAYER 1: RAW IMMUTABLE DOCUMENTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw_documents (
    -- Identity
    raw_doc_id          VARCHAR PRIMARY KEY,   -- sha256(source + url + ingested_at)
    content_hash        VARCHAR NOT NULL,       -- sha256(body) for dedup

    -- Provenance
    source              VARCHAR NOT NULL,       -- 'gdelt' | 'rss' | 'metaculus' | 'acled'
                                               -- | 'official' | 'court' | 'ticketing' | 'market'
    source_url          VARCHAR,
    source_feed         VARCHAR,               -- RSS feed URL, GDELT query, etc.

    -- Timestamps (never estimated — use NULL if unknown)
    published_at        TIMESTAMPTZ,           -- as reported by source
    ingested_at         TIMESTAMPTZ NOT NULL,  -- when we downloaded it

    -- Content
    title               VARCHAR,
    body                TEXT,
    language            VARCHAR(5),            -- BCP-47, e.g. 'en', 'es', 'ar'

    -- Source metadata
    source_quality      FLOAT,                 -- 0-1, calibrated per source type
    is_official_source  BOOLEAN DEFAULT FALSE, -- state gazette, UN, court filing, etc.

    -- Scraping metadata
    http_status         SMALLINT,
    collector_version   VARCHAR,               -- version of collector that fetched this

    -- Partitioning helper (denormalized for query speed; set on insert)
    published_date      DATE,                  -- CAST(published_at AS DATE)

    CONSTRAINT raw_doc_content_hash_ingested UNIQUE (content_hash, source)
);

CREATE INDEX IF NOT EXISTS idx_raw_docs_source_date
    ON raw_documents (source, published_at);

CREATE INDEX IF NOT EXISTS idx_raw_docs_hash
    ON raw_documents (content_hash);


-- =============================================================================
-- LAYER 2: CANONICAL DOCUMENTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS canonical_documents (
    -- Identity
    doc_id              VARCHAR PRIMARY KEY,   -- derived from raw_doc_id
    raw_doc_id          VARCHAR NOT NULL REFERENCES raw_documents(raw_doc_id),

    -- Canonical fields
    source              VARCHAR NOT NULL,
    published_at        TIMESTAMPTZ NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL,
    title               VARCHAR,
    body_clean          TEXT,                  -- normalized, stripped HTML
    language            VARCHAR(5),

    -- Quality
    source_quality      FLOAT,
    is_official_source  BOOLEAN DEFAULT FALSE,
    dedup_cluster_id    VARCHAR,               -- groups near-duplicate docs

    -- Tags (extracted by normalizer)
    country_tags        VARCHAR[],             -- ISO-3166 country codes
    entity_mentions     VARCHAR[],             -- raw mention strings before linking
    topic_tags          VARCHAR[],             -- 'conflict' | 'politics' | 'legal' | ...

    -- Contradiction flags
    contradicts_doc_ids VARCHAR[],             -- other docs that contradict this one
    contradiction_score FLOAT DEFAULT 0.0,    -- 0=no contradiction, 1=fully contradicted

    -- Processor metadata
    normalizer_version  VARCHAR,
    processed_at        TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_canonical_docs_published
    ON canonical_documents (published_at);

CREATE INDEX IF NOT EXISTS idx_canonical_docs_source
    ON canonical_documents (source, published_at);

-- Note: array columns (country_tags, entity_mentions) are filtered in WHERE clauses
-- using list_contains(). DuckDB 1.x does not support GIN-style array indexes.


-- =============================================================================
-- LAYER 3a: ENTITIES (canonical registry)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entities (
    entity_id           VARCHAR PRIMARY KEY,   -- stable UUID per real-world entity
    canonical_name      VARCHAR NOT NULL,      -- "Pedro Sánchez", "Ukraine", "Bad Bunny"
    entity_type         VARCHAR NOT NULL,      -- 'person' | 'country' | 'organization'
                                               -- | 'artist' | 'venue' | 'process' | 'market'

    -- Disambiguation
    wikidata_id         VARCHAR,               -- Q-number for grounding
    country             VARCHAR,               -- ISO-3166
    description         VARCHAR,               -- short disambiguation string

    -- Structural priors (from features/country_data.py for country entities)
    conflict_baserate   FLOAT,                 -- UCDP fraction years with conflict
    polity_norm         FLOAT,                 -- Polity5 / 10
    mil_spending_norm   FLOAT,                 -- SIPRI mil% GDP / 10

    -- Lifecycle
    first_seen_at       TIMESTAMPTZ,
    last_updated_at     TIMESTAMPTZ,
    is_active           BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_entities_type
    ON entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_entities_canonical_name
    ON entities (canonical_name);


-- Entity aliases: "Sánchez", "Pedro Sánchez", "PM of Spain" → same entity_id
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id            VARCHAR PRIMARY KEY,
    entity_id           VARCHAR NOT NULL REFERENCES entities(entity_id),
    alias               VARCHAR NOT NULL,
    alias_language      VARCHAR(5) DEFAULT 'en',
    source              VARCHAR,               -- 'wikidata' | 'manual' | 'extracted'
    confidence          FLOAT DEFAULT 1.0,

    CONSTRAINT entity_alias_unique UNIQUE (alias, alias_language)
);

CREATE INDEX IF NOT EXISTS idx_aliases_alias
    ON entity_aliases (alias);


-- Which entities appear in which document
CREATE TABLE IF NOT EXISTS document_entities (
    doc_id              VARCHAR NOT NULL REFERENCES canonical_documents(doc_id),
    entity_id           VARCHAR NOT NULL REFERENCES entities(entity_id),
    mention_text        VARCHAR NOT NULL,      -- original surface form
    mention_role        VARCHAR,               -- 'subject' | 'object' | 'location' | 'other'
    confidence          FLOAT DEFAULT 1.0,
    extractor_version   VARCHAR,

    PRIMARY KEY (doc_id, entity_id, mention_role)
);

CREATE INDEX IF NOT EXISTS idx_doc_entities_entity
    ON document_entities (entity_id);


-- =============================================================================
-- LAYER 3b: CANONICAL EVENTS
-- =============================================================================
-- Each document can generate 0..N events.
-- Event types span all domains (conflict, political, legal, entertainment, economic).

CREATE TABLE IF NOT EXISTS canonical_events (
    event_id            VARCHAR PRIMARY KEY,
    doc_id              VARCHAR NOT NULL REFERENCES canonical_documents(doc_id),

    -- Timing
    event_time          TIMESTAMPTZ NOT NULL,  -- when the event occurred (not published)
    event_time_precision VARCHAR DEFAULT 'day', -- 'second' | 'hour' | 'day' | 'month' | 'year'

    -- Taxonomy (matches question.parser event families and predicates)
    event_type          VARCHAR NOT NULL,
    -- Examples:
    --   conflict:      'military_action', 'ceasefire_signal', 'coup_attempt', 'nuclear_test'
    --   political:     'resignation_signal', 'election_called', 'sanction_imposed',
    --                  'coalition_collapse', 'government_formed'
    --   legal:         'arrest_signal', 'indictment', 'extradition_request',
    --                  'trial_started', 'verdict'
    --   entertainment: 'tour_announcement', 'venue_booking', 'album_release',
    --                  'concert_confirmed', 'artist_cancellation'
    --   economic:      'rate_decision', 'gdp_release', 'default_signal',
    --                  'ipo_filing', 'merger_announced'

    -- Primary actors
    actor_entity_ids    VARCHAR[],             -- entity_ids of subjects
    target_entity_ids   VARCHAR[],             -- entity_ids of objects/targets
    location_entity_id  VARCHAR REFERENCES entities(entity_id),

    -- Event attributes
    intensity           FLOAT,                 -- 0-10, source-normalized
    polarity            FLOAT,                 -- -1 (negative) to +1 (positive)
    certainty           FLOAT,                 -- 0-1: how certain is the event?
    is_official         BOOLEAN DEFAULT FALSE, -- from official source?
    fatalities          INTEGER DEFAULT 0,

    -- Source quality
    source_quality      FLOAT,
    independent_sources SMALLINT DEFAULT 1,    -- # distinct sources confirming this
    contradiction_score FLOAT DEFAULT 0.0,

    -- Extractor metadata
    extractor_version   VARCHAR NOT NULL,      -- version that produced this event
    extraction_method   VARCHAR,               -- 'rule' | 'ner' | 'classifier' | 'manual'
    confidence          FLOAT DEFAULT 1.0,

    -- Partitioning helper (set on insert: CAST(event_time AS DATE))
    event_date          DATE
);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON canonical_events (event_type, event_time);

CREATE INDEX IF NOT EXISTS idx_events_doc
    ON canonical_events (doc_id);

-- array actor_entity_ids: use list_contains(actor_entity_ids, ?) in queries

CREATE INDEX IF NOT EXISTS idx_events_location
    ON canonical_events (location_entity_id, event_time);


-- Event arguments: structured slot-filler representation
CREATE TABLE IF NOT EXISTS event_arguments (
    event_id            VARCHAR NOT NULL REFERENCES canonical_events(event_id),
    role                VARCHAR NOT NULL,      -- 'agent', 'patient', 'instrument', 'location'
    entity_id           VARCHAR REFERENCES entities(entity_id),
    freetext_value      VARCHAR,               -- for non-entity arguments
    confidence          FLOAT DEFAULT 1.0,

    PRIMARY KEY (event_id, role)
);


-- =============================================================================
-- LAYER 4: TEMPORAL KNOWLEDGE GRAPH (edge table)
-- =============================================================================
-- Stores relations as temporal edges with validity intervals.
-- First implement as tables; add graph DB layer only when needed.

CREATE TABLE IF NOT EXISTS entity_relations_temporal (
    relation_id         VARCHAR PRIMARY KEY,

    -- Endpoints
    subject_entity_id   VARCHAR NOT NULL REFERENCES entities(entity_id),
    object_entity_id    VARCHAR NOT NULL REFERENCES entities(entity_id),

    -- Relation type
    relation_type       VARCHAR NOT NULL,
    -- Examples:
    --   'holds_office'          -- person → position
    --   'member_of'             -- person → organization/party
    --   'governs'               -- person/party → country
    --   'under_investigation'   -- person → investigation
    --   'sanctioned_by'         -- entity → sanctioning body
    --   'conflicts_with'        -- country → country
    --   'negotiates_with'       -- entity → entity
    --   'allied_with'           -- entity → entity
    --   'announced_event'       -- artist → venue/event
    --   'scheduled_at'          -- event → venue
    --   'performed_in'          -- artist → country
    --   'indicted_by'           -- person → court/prosecutor
    --   'arrested_by'           -- person → authority

    -- Temporal validity
    valid_from          TIMESTAMPTZ NOT NULL,
    valid_to            TIMESTAMPTZ,           -- NULL = still valid

    -- Evidence
    source_doc_ids      VARCHAR[],             -- docs supporting this relation
    source_event_ids    VARCHAR[],             -- events supporting this relation
    confidence          FLOAT DEFAULT 1.0,
    is_official         BOOLEAN DEFAULT FALSE,

    -- Attributes (relation-specific metadata)
    attributes          JSON,                  -- {"position": "Prime Minister", "country": "Spain"}

    -- Versioning
    asserted_at         TIMESTAMPTZ NOT NULL,  -- when we learned this
    retracted_at        TIMESTAMPTZ,           -- when we learned it ended
    extractor_version   VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_relations_subject_type
    ON entity_relations_temporal (subject_entity_id, relation_type, valid_from);

CREATE INDEX IF NOT EXISTS idx_relations_object_type
    ON entity_relations_temporal (object_entity_id, relation_type, valid_from);

CREATE INDEX IF NOT EXISTS idx_relations_valid
    ON entity_relations_temporal (relation_type, valid_from, valid_to);


-- =============================================================================
-- LAYER 5: FEATURE STORE (temporal, reproducible)
-- =============================================================================
-- Every feature vector used for training or inference is stored here.
-- Enables exact "as of time T" reconstruction and audit.

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id         VARCHAR PRIMARY KEY,

    -- Context
    question_id         VARCHAR,               -- NULL for training windows without question
    subject_entity_id   VARCHAR REFERENCES entities(entity_id),
    as_of_time          TIMESTAMPTZ NOT NULL,  -- cutoff: no data after this was used

    -- Question context (for inference)
    question_predicate  VARCHAR,               -- e.g. 'resign', 'military_escalation'
    question_deadline   DATE,

    -- Feature vector (stored as JSON for flexibility across schema versions)
    explicit_features   JSON NOT NULL,         -- the flat dict the model sees
    feature_schema_ver  VARCHAR NOT NULL,       -- 'v3' etc. — must match model version

    -- References to source data used
    doc_ids_used        VARCHAR[],             -- canonical docs in the window
    event_ids_used      VARCHAR[],             -- events in the window
    window_start        TIMESTAMPTZ,
    window_end          TIMESTAMPTZ,

    -- Latent features (stored by reference, not inline)
    embedding_id        VARCHAR,               -- → embedding_registry

    -- Build metadata
    builder_version     VARCHAR NOT NULL,
    built_at            TIMESTAMPTZ NOT NULL,

    -- Outcome (filled post-resolution for training)
    outcome             SMALLINT,              -- 0 | 1 | NULL (unresolved)
    outcome_resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_snapshots_subject_time
    ON feature_snapshots (subject_entity_id, as_of_time);

CREATE INDEX IF NOT EXISTS idx_snapshots_predicate
    ON feature_snapshots (question_predicate, as_of_time);

CREATE INDEX IF NOT EXISTS idx_snapshots_schema_ver
    ON feature_snapshots (feature_schema_ver);

CREATE INDEX IF NOT EXISTS idx_snapshots_outcome
    ON feature_snapshots (outcome, question_predicate);


-- Embedding registry: stores references to vector files, not vectors inline
CREATE TABLE IF NOT EXISTS embedding_registry (
    embedding_id        VARCHAR PRIMARY KEY,
    subject_entity_id   VARCHAR REFERENCES entities(entity_id),
    as_of_time          TIMESTAMPTZ NOT NULL,
    embedding_type      VARCHAR NOT NULL,      -- 'entity' | 'document' | 'subgraph' | 'sequence'
    model_name          VARCHAR NOT NULL,
    dimension           SMALLINT NOT NULL,
    storage_path        VARCHAR NOT NULL,      -- path to .npy or vector DB reference
    built_at            TIMESTAMPTZ NOT NULL
);


-- =============================================================================
-- LAYER 6: QUESTION & RESOLUTION REGISTRY
-- =============================================================================

CREATE TABLE IF NOT EXISTS question_templates (
    template_id         VARCHAR PRIMARY KEY,
    predicate           VARCHAR NOT NULL,      -- canonical verb: 'resign', 'military_escalation'
    event_family        VARCHAR NOT NULL,      -- 'conflict' | 'political' | 'legal' | ...
    subject_type        VARCHAR NOT NULL,      -- 'person' | 'country' | 'artist' | ...

    -- Resolution guidance
    resolution_rule_template VARCHAR,          -- "Formal resignation of {subject} from {office}"
    typical_horizon_days     SMALLINT,         -- typical question horizon

    -- Model routing
    model_id            VARCHAR,               -- which model handles this predicate
    requires_features   VARCHAR[],             -- which feature families are needed

    -- Data requirements
    required_sources    VARCHAR[],             -- which collectors needed
    notes               VARCHAR
);


CREATE TABLE IF NOT EXISTS questions (
    question_id         VARCHAR PRIMARY KEY,
    template_id         VARCHAR REFERENCES question_templates(template_id),

    -- Natural language
    raw_text            VARCHAR NOT NULL,

    -- Structured parse (from question.parser)
    subject             VARCHAR NOT NULL,
    subject_entity_id   VARCHAR REFERENCES entities(entity_id),
    predicate           VARCHAR NOT NULL,
    event_family        VARCHAR NOT NULL,
    jurisdiction        VARCHAR,
    is_negated          BOOLEAN DEFAULT FALSE,

    -- Resolution
    deadline            DATE NOT NULL,
    resolution_rule     VARCHAR NOT NULL,

    -- Status
    status              VARCHAR DEFAULT 'open', -- 'open' | 'resolved' | 'void' | 'ood'

    -- OOD assessment
    ood_score           FLOAT,
    matched_model       VARCHAR,
    parse_confidence    FLOAT,

    -- Metadata
    created_at          TIMESTAMPTZ NOT NULL,
    as_of_time          TIMESTAMPTZ NOT NULL,  -- data cutoff at question creation
    source              VARCHAR DEFAULT 'user' -- 'user' | 'metaculus' | 'polymarket' | 'synthetic'
);

CREATE INDEX IF NOT EXISTS idx_questions_status
    ON questions (status, deadline);

CREATE INDEX IF NOT EXISTS idx_questions_predicate
    ON questions (predicate, event_family);

CREATE INDEX IF NOT EXISTS idx_questions_subject
    ON questions (subject_entity_id);


CREATE TABLE IF NOT EXISTS question_resolutions (
    resolution_id       VARCHAR PRIMARY KEY,
    question_id         VARCHAR NOT NULL REFERENCES questions(question_id),

    -- Outcome
    outcome             SMALLINT NOT NULL,     -- 0 | 1
    resolved_at         TIMESTAMPTZ NOT NULL,  -- when we determined the outcome
    deadline_was        DATE NOT NULL,         -- the deadline that applied

    -- Evidence
    resolver_source     VARCHAR NOT NULL,      -- 'official_gazette' | 'news' | 'metaculus' | 'manual'
    resolver_doc_ids    VARCHAR[],
    resolution_notes    VARCHAR,

    -- Quality
    resolution_confidence FLOAT DEFAULT 1.0,
    is_ambiguous        BOOLEAN DEFAULT FALSE,
    ambiguity_notes     VARCHAR,

    -- Audit
    resolved_by         VARCHAR DEFAULT 'system'
);


-- Predictions: every prediction made, permanently logged
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       VARCHAR PRIMARY KEY,   -- UUID
    question_id         VARCHAR REFERENCES questions(question_id),
    snapshot_id         VARCHAR REFERENCES feature_snapshots(snapshot_id),

    -- Prediction values
    raw_prob            FLOAT NOT NULL,
    calibrated_prob     FLOAT NOT NULL,
    ci_lo               FLOAT,
    ci_hi               FLOAT,
    ci_method           VARCHAR,
    answer              VARCHAR,               -- 'YES' | 'NO'

    -- Model
    model_id            VARCHAR NOT NULL,
    model_version       VARCHAR,
    schema_version      VARCHAR,
    market_override     BOOLEAN DEFAULT FALSE,
    market_prob_used    FLOAT,

    -- Attribution summary (top features)
    top_features        JSON,
    flip_set_size       SMALLINT,

    -- Timing
    predicted_at        TIMESTAMPTZ NOT NULL,
    as_of_time          TIMESTAMPTZ NOT NULL,

    -- Outcome (filled after resolution)
    brier_component     FLOAT,                 -- (p - y)^2 for this prediction
    was_correct         BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_predictions_question
    ON predictions (question_id, predicted_at);

CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON predictions (model_id, predicted_at);


-- =============================================================================
-- LAYER 7: DATA LINEAGE (the most important table almost nobody builds)
-- =============================================================================
-- Tracks every transformation: raw doc → canonical → event → feature → prediction.
-- Without this, auditing failures is guesswork.

CREATE TABLE IF NOT EXISTS data_lineage (
    lineage_id          VARCHAR PRIMARY KEY,

    -- What was produced
    output_type         VARCHAR NOT NULL,
    -- 'canonical_document' | 'entity' | 'entity_alias' | 'canonical_event'
    -- | 'entity_relation' | 'feature_snapshot' | 'embedding' | 'prediction'
    output_id           VARCHAR NOT NULL,      -- ID of the produced artifact

    -- What it came from
    input_type          VARCHAR NOT NULL,
    input_ids           VARCHAR[] NOT NULL,    -- IDs of source artifacts

    -- How it was produced
    processor_name      VARCHAR NOT NULL,      -- e.g. 'normalizer.canonical', 'features.builder'
    processor_version   VARCHAR NOT NULL,
    processor_config    JSON,                  -- config / hyperparams used

    -- When
    processed_at        TIMESTAMPTZ NOT NULL,
    duration_ms         INTEGER,

    -- Quality
    confidence          FLOAT,
    warnings            VARCHAR[],
    errors              VARCHAR[],

    -- Reproducibility
    input_hashes        VARCHAR[],             -- content hashes of inputs at processing time
    is_deterministic    BOOLEAN DEFAULT TRUE   -- whether rerunning gives same output
);

CREATE INDEX IF NOT EXISTS idx_lineage_output
    ON data_lineage (output_type, output_id);

CREATE INDEX IF NOT EXISTS idx_lineage_input
    ON data_lineage (input_type, processed_at);

CREATE INDEX IF NOT EXISTS idx_lineage_processor
    ON data_lineage (processor_name, processed_at);


-- =============================================================================
-- VIEWS: common query patterns as named views
-- =============================================================================

-- Active relations as of now
CREATE OR REPLACE VIEW active_relations AS
SELECT *
FROM entity_relations_temporal
WHERE valid_from <= current_timestamp
  AND (valid_to IS NULL OR valid_to > current_timestamp);


-- Unresolved questions past deadline (needs resolution)
CREATE OR REPLACE VIEW overdue_questions AS
SELECT q.*, qr.outcome
FROM questions q
LEFT JOIN question_resolutions qr ON q.question_id = qr.question_id
WHERE q.status = 'open'
  AND q.deadline < CURRENT_DATE
  AND qr.resolution_id IS NULL;


-- Training-ready feature snapshots (have outcome, have features, valid schema)
CREATE OR REPLACE VIEW training_ready_snapshots AS
SELECT
    fs.*,
    qr.outcome,
    qr.resolved_at,
    q.predicate,
    q.event_family,
    q.deadline
FROM feature_snapshots fs
JOIN questions q ON fs.question_id = q.question_id
JOIN question_resolutions qr ON q.question_id = qr.question_id
WHERE fs.outcome IS NOT NULL
  AND fs.explicit_features IS NOT NULL
  AND qr.is_ambiguous = FALSE;


-- Calibration audit: predictions vs outcomes grouped by probability bin
CREATE OR REPLACE VIEW calibration_audit AS
SELECT
    model_id,
    schema_version,
    ROUND(calibrated_prob * 10) / 10 AS prob_bin,
    COUNT(*) AS n,
    AVG(calibrated_prob) AS mean_pred,
    AVG(CAST(was_correct AS FLOAT)) AS mean_actual,
    AVG(brier_component) AS mean_brier
FROM predictions
WHERE was_correct IS NOT NULL
GROUP BY model_id, schema_version, prob_bin
ORDER BY model_id, prob_bin;


-- Entity activity timeline: last N events per entity
CREATE OR REPLACE VIEW entity_recent_events AS
SELECT
    e.entity_id,
    e.canonical_name,
    e.entity_type,
    ce.event_id,
    ce.event_type,
    ce.event_time,
    ce.intensity,
    ce.polarity,
    ce.confidence,
    ROW_NUMBER() OVER (PARTITION BY e.entity_id ORDER BY ce.event_time DESC) AS recency_rank
FROM entities e
JOIN canonical_events ce ON list_contains(ce.actor_entity_ids, e.entity_id)
WHERE ce.event_time >= current_timestamp - INTERVAL '90 days';


-- =============================================================================
-- LAYER 8 — World Model State
-- Persistent entity state updated daily by scripts/update_world_state.py.
-- This is the foundation of the large world model.
-- =============================================================================

CREATE TABLE IF NOT EXISTS world_state (
    -- Identity
    state_id            VARCHAR PRIMARY KEY,   -- sha256(entity_id || as_of_date)
    entity_id           VARCHAR NOT NULL,
    as_of_date          DATE NOT NULL,

    -- Conflict / security features
    military_count_7d       FLOAT NOT NULL DEFAULT 0.0,
    military_count_30d      FLOAT NOT NULL DEFAULT 0.0,
    protest_count_7d        FLOAT NOT NULL DEFAULT 0.0,
    protest_count_30d       FLOAT NOT NULL DEFAULT 0.0,
    diplomatic_count_7d     FLOAT NOT NULL DEFAULT 0.0,
    ceasefire_count_7d      FLOAT NOT NULL DEFAULT 0.0,
    sanction_count_7d       FLOAT NOT NULL DEFAULT 0.0,
    military_intensity_7d   FLOAT NOT NULL DEFAULT 0.0,
    protest_intensity_7d    FLOAT NOT NULL DEFAULT 0.0,
    overall_intensity_7d    FLOAT NOT NULL DEFAULT 0.0,
    military_accel          FLOAT NOT NULL DEFAULT 1.0,
    protest_accel           FLOAT NOT NULL DEFAULT 1.0,
    overall_accel           FLOAT NOT NULL DEFAULT 1.0,
    avg_polarity_7d         FLOAT NOT NULL DEFAULT 0.0,
    avg_polarity_30d        FLOAT NOT NULL DEFAULT 0.0,
    tone_trend              FLOAT NOT NULL DEFAULT 0.0,
    source_diversity_7d     FLOAT NOT NULL DEFAULT 0.0,
    avg_independent_sources FLOAT NOT NULL DEFAULT 1.0,
    has_military_7d         FLOAT NOT NULL DEFAULT 0.0,
    has_ceasefire_7d        FLOAT NOT NULL DEFAULT 0.0,
    escalation_index        FLOAT NOT NULL DEFAULT 0.0,
    ceasefire_ratio_7d      FLOAT NOT NULL DEFAULT 0.0,
    event_velocity_7d       FLOAT NOT NULL DEFAULT 1.0,
    military_share_7d       FLOAT NOT NULL DEFAULT 0.0,

    -- Political domain features
    pol_resignation_signals FLOAT NOT NULL DEFAULT 0.0,
    pol_approval_pressure   FLOAT NOT NULL DEFAULT 0.0,
    pol_coalition_stability FLOAT NOT NULL DEFAULT 0.5,
    pol_electoral_proximity FLOAT NOT NULL DEFAULT 0.0,
    pol_judicial_pressure   FLOAT NOT NULL DEFAULT 0.0,

    -- Economic domain features
    eco_rate_change_prob    FLOAT NOT NULL DEFAULT 0.0,
    eco_gdp_momentum        FLOAT NOT NULL DEFAULT 0.0,
    eco_debt_stress         FLOAT NOT NULL DEFAULT 0.0,
    eco_market_volatility   FLOAT NOT NULL DEFAULT 0.0,
    eco_policy_uncertainty  FLOAT NOT NULL DEFAULT 0.0,

    -- Structural features (from country_data / WB / V-Dem)
    country_conflict_baserate FLOAT NOT NULL DEFAULT 0.15,
    country_polity_norm       FLOAT NOT NULL DEFAULT 0.0,
    country_mil_spending_norm FLOAT NOT NULL DEFAULT 0.15,
    wgi_pol_stability         FLOAT,
    wgi_gov_effectiveness     FLOAT,
    wgi_rule_of_law           FLOAT,
    fred_vix                  FLOAT,
    fred_yield_spread         FLOAT,

    -- Transition metadata
    smoothing_alpha         FLOAT NOT NULL DEFAULT 0.3,
    causal_inflow           JSON,
    n_events_used           INTEGER NOT NULL DEFAULT 0,
    data_completeness       FLOAT NOT NULL DEFAULT 1.0,
    sources_used            VARCHAR[],
    updater_version         VARCHAR NOT NULL DEFAULT 'v1',

    -- Validity window
    computed_at             TIMESTAMPTZ NOT NULL,
    valid_from              TIMESTAMPTZ NOT NULL,
    valid_to                TIMESTAMPTZ,

    CONSTRAINT world_state_entity_date UNIQUE (entity_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_world_state_entity_date
    ON world_state (entity_id, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_world_state_date
    ON world_state (as_of_date DESC);

-- Append-only history for time-series analysis and backtesting
CREATE TABLE IF NOT EXISTS world_state_history (
    history_id          VARCHAR PRIMARY KEY,
    entity_id           VARCHAR NOT NULL,
    as_of_date          DATE NOT NULL,
    computed_at         TIMESTAMPTZ NOT NULL,
    features            JSON NOT NULL,
    delta_from_prior    JSON,
    prior_state_date    DATE,
    causal_inflow_summary JSON,
    sources_used        VARCHAR[],
    n_events_used       INTEGER NOT NULL DEFAULT 0,
    data_completeness   FLOAT NOT NULL DEFAULT 1.0,
    updater_version     VARCHAR NOT NULL DEFAULT 'v1'
);

CREATE INDEX IF NOT EXISTS idx_wsh_entity_date
    ON world_state_history (entity_id, as_of_date DESC);

-- Current world state view (latest valid row per entity)
CREATE OR REPLACE VIEW current_world_state AS
SELECT ws.*
FROM world_state ws
INNER JOIN (
    SELECT entity_id, MAX(as_of_date) AS max_date
    FROM world_state
    WHERE valid_to IS NULL
    GROUP BY entity_id
) latest ON ws.entity_id = latest.entity_id AND ws.as_of_date = latest.max_date
WHERE ws.valid_to IS NULL;
