import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const databaseName = process.env.D1_DATABASE_NAME?.trim() || "qwen-tts-db";
const useLocal = process.argv.includes("--local");

const baseArgs = ["wrangler", "d1", "execute", databaseName];
if (useLocal) {
  baseArgs.push("--local");
} else {
  baseArgs.push("--remote");
}

function runWrangler(args, label) {
  try {
    return execFileSync("npx", [...baseArgs, ...args], {
      cwd: projectRoot,
      encoding: "utf8",
      stdio: ["inherit", "pipe", "pipe"],
    });
  } catch (error) {
    const stdout =
      error && typeof error === "object" && "stdout" in error ? String(error.stdout ?? "") : "";
    const stderr =
      error && typeof error === "object" && "stderr" in error ? String(error.stderr ?? "") : "";
    const detail = [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
    throw new Error(detail ? `${label} failed:\n${detail}` : `${label} failed`);
  }
}

function execFile(file) {
  runWrangler(["--file", file], `Applying ${file}`);
}

function query(sql) {
  const output = runWrangler(["--command", sql, "--json"], `Executing query: ${sql}`);
  const parsed = JSON.parse(output);
  const result = Array.isArray(parsed) ? parsed[0] : parsed;
  if (!result || result.success !== true) {
    throw new Error(`Query failed: ${sql}`);
  }
  return Array.isArray(result.results) ? result.results : [];
}

function execute(sql) {
  query(sql);
}

function getColumns(tableName) {
  return new Set(query(`PRAGMA table_info(${tableName});`).map((row) => String(row.name)));
}

function ensureColumn(tableName, columns, columnName, definition) {
  if (columns.has(columnName)) {
    return;
  }
  execute(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
  columns.add(columnName);
  console.log(`Added ${tableName}.${columnName}`);
}

function applySchemaWithCompatibility() {
  try {
    execFile("schema.sql");
    return;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const lower = message.toLowerCase();
    const missingCampaignColumns =
      lower.includes("campaign_id") || lower.includes("attempt_index") || lower.includes("training_jobs");

    if (!missingCampaignColumns) {
      throw error;
    }

    console.warn("schema.sql failed before backfill; attempting compatibility backfill for training_jobs campaign columns");
    const compatibilityColumns = getColumns("training_jobs");
    ensureColumn("training_jobs", compatibilityColumns, "campaign_id", "TEXT");
    ensureColumn("training_jobs", compatibilityColumns, "attempt_index", "INTEGER");
    execFile("schema.sql");
  }
}

applySchemaWithCompatibility();

const trainingJobColumns = getColumns("training_jobs");
ensureColumn("training_jobs", trainingJobColumns, "last_heartbeat_at", "INTEGER");
ensureColumn("training_jobs", trainingJobColumns, "summary_json", "TEXT DEFAULT '{}'");
ensureColumn("training_jobs", trainingJobColumns, "metrics_json", "TEXT DEFAULT '{}'");
ensureColumn("training_jobs", trainingJobColumns, "log_r2_prefix", "TEXT");
ensureColumn("training_jobs", trainingJobColumns, "job_token", "TEXT");
ensureColumn("training_jobs", trainingJobColumns, "round_id", "TEXT");
ensureColumn("training_jobs", trainingJobColumns, "dataset_snapshot_id", "TEXT");
ensureColumn("training_jobs", trainingJobColumns, "supervisor_json", "TEXT DEFAULT '{}'");
ensureColumn("training_jobs", trainingJobColumns, "campaign_id", "TEXT");
ensureColumn("training_jobs", trainingJobColumns, "attempt_index", "INTEGER");
execute(
  "CREATE UNIQUE INDEX IF NOT EXISTS idx_training_jobs_campaign_attempt_unique ON training_jobs(campaign_id, attempt_index) WHERE campaign_id IS NOT NULL AND attempt_index IS NOT NULL"
);

const voiceColumns = getColumns("voices");
ensureColumn("voices", voiceColumns, "candidate_checkpoint_r2_prefix", "TEXT");
ensureColumn("voices", voiceColumns, "candidate_run_name", "TEXT");
ensureColumn("voices", voiceColumns, "candidate_epoch", "INTEGER");
ensureColumn("voices", voiceColumns, "candidate_score", "REAL");
ensureColumn("voices", voiceColumns, "candidate_job_id", "TEXT");
ensureColumn("voices", voiceColumns, "active_round_id", "TEXT");
ensureColumn("voices", voiceColumns, "checkpoint_preset", "TEXT");
ensureColumn("voices", voiceColumns, "checkpoint_score", "REAL");
ensureColumn("voices", voiceColumns, "checkpoint_job_id", "TEXT");
ensureColumn("voices", voiceColumns, "candidate_preset", "TEXT");

const trainingRoundColumns = getColumns("training_rounds");
ensureColumn("training_rounds", trainingRoundColumns, "production_preset", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "production_score", "REAL");
ensureColumn("training_rounds", trainingRoundColumns, "production_job_id", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "champion_checkpoint_r2_prefix", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "champion_run_name", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "champion_epoch", "INTEGER");
ensureColumn("training_rounds", trainingRoundColumns, "champion_preset", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "champion_score", "REAL");
ensureColumn("training_rounds", trainingRoundColumns, "champion_job_id", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "selected_checkpoint_r2_prefix", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "selected_run_name", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "selected_epoch", "INTEGER");
ensureColumn("training_rounds", trainingRoundColumns, "selected_preset", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "selected_score", "REAL");
ensureColumn("training_rounds", trainingRoundColumns, "selected_job_id", "TEXT");
ensureColumn("training_rounds", trainingRoundColumns, "adoption_mode", "TEXT");

console.log(`D1 schema is up to date${useLocal ? " (local)" : ""}.`);
