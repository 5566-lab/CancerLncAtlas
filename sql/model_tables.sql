CREATE TABLE IF NOT EXISTS model_run (
  analysis_version TEXT PRIMARY KEY,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  status TEXT,
  config_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS model_prediction (
  candidate_id TEXT,
  cancer_id TEXT,
  lncrna_id TEXT,
  pathway_family_id TEXT,
  model_name TEXT,
  fold_id TEXT,
  calibrated_probability DOUBLE PRECISION,
  uncertainty DOUBLE PRECISION,
  predicted_direction TEXT,
  observed_evidence_score DOUBLE PRECISION,
  analysis_version TEXT,
  PRIMARY KEY(candidate_id, model_name, analysis_version)
);

CREATE TABLE IF NOT EXISTS model_geneset (
  geneset_id TEXT PRIMARY KEY,
  geneset_name TEXT,
  cancer_id TEXT,
  pathway_family_id TEXT,
  direction TEXT,
  member_count INTEGER,
  analysis_version TEXT
);

CREATE TABLE IF NOT EXISTS model_geneset_member (
  geneset_id TEXT,
  lncrna_id TEXT,
  member_class TEXT,
  member_weight DOUBLE PRECISION,
  rank INTEGER,
  calibrated_probability DOUBLE PRECISION,
  observed_evidence_score DOUBLE PRECISION,
  PRIMARY KEY(geneset_id, lncrna_id)
);

-- V3.0 clinical extension tables. Survival endpoints remain separate from
-- functional relationship confidence and discovery probabilities.
CREATE TABLE IF NOT EXISTS clinical_endpoint_summary (
  cancer_id TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  n_patients INTEGER,
  n_events INTEGER,
  c_index DOUBLE PRECISION,
  c_index_sd DOUBLE PRECISION,
  PRIMARY KEY (cancer_id, endpoint)
);

CREATE TABLE IF NOT EXISTS lncrna_clinical_association (
  cancer_id TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  subject_type TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  modifier_id TEXT,
  hazard_ratio DOUBLE PRECISION,
  ci_lower DOUBLE PRECISION,
  ci_upper DOUBLE PRECISION,
  p_value DOUBLE PRECISION,
  fdr DOUBLE PRECISION,
  replication_tier TEXT,
  clinical_survival_probability DOUBLE PRECISION,
  model_version TEXT,
  PRIMARY KEY (cancer_id, endpoint, subject_type, subject_id, modifier_id)
);
