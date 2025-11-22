-- ASR Training Database Schema
-- This schema manages audio files, transcripts, and training experiments

-- Table: AudioFiles
-- Stores metadata about audio recordings
CREATE TABLE IF NOT EXISTS AudioFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    sample_rate INTEGER,
    channels INTEGER,
    format TEXT,
    language TEXT DEFAULT 'vi',
    speaker_id TEXT,
    audio_quality TEXT CHECK(audio_quality IN ('high', 'medium', 'low')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table: Transcripts
-- Stores ground truth transcriptions for audio files
CREATE TABLE IF NOT EXISTS Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER NOT NULL,
    transcript TEXT NOT NULL,
    normalized_transcript TEXT,
    word_count INTEGER,
    char_count INTEGER,
    annotation_type TEXT DEFAULT 'manual' CHECK(annotation_type IN ('manual', 'auto', 'corrected')),
    confidence_score REAL,
    annotator_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(audio_file_id) REFERENCES AudioFiles(id) ON DELETE CASCADE
);

-- Table: DataSplits
-- Tracks train/validation/test splits
CREATE TABLE IF NOT EXISTS DataSplits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER NOT NULL,
    split_type TEXT NOT NULL CHECK(split_type IN ('train', 'val', 'test')),
    split_version TEXT DEFAULT 'v1',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(audio_file_id) REFERENCES AudioFiles(id) ON DELETE CASCADE,
    UNIQUE(audio_file_id, split_version)
);

-- Table: Models
-- Registry of trained models
CREATE TABLE IF NOT EXISTS Models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    architecture TEXT,
    version TEXT,
    checkpoint_path TEXT,
    config_json TEXT,
    total_parameters INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table: TrainingRuns
-- Tracks individual training experiments
CREATE TABLE IF NOT EXISTS TrainingRuns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    run_name TEXT NOT NULL UNIQUE,
    description TEXT,
    config_json TEXT,
    split_version TEXT DEFAULT 'v1',
    
    -- Hyperparameters
    batch_size INTEGER,
    learning_rate REAL,
    num_epochs INTEGER,
    optimizer TEXT,
    
    -- Results
    final_train_loss REAL,
    final_val_loss REAL,
    best_val_loss REAL,
    best_epoch INTEGER,
    word_error_rate REAL,
    character_error_rate REAL,
    
    -- Training metadata
    total_training_time_seconds REAL,
    gpu_name TEXT,
    num_gpus INTEGER DEFAULT 1,
    status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed', 'stopped')),
    
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    
    FOREIGN KEY(model_id) REFERENCES Models(id) ON DELETE SET NULL
);

-- Table: EpochMetrics
-- Detailed metrics per epoch
CREATE TABLE IF NOT EXISTS EpochMetrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL,
    val_loss REAL,
    learning_rate REAL,
    wer REAL,
    cer REAL,
    epoch_time_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(training_run_id) REFERENCES TrainingRuns(id) ON DELETE CASCADE,
    UNIQUE(training_run_id, epoch)
);

-- Table: Predictions
-- Store model predictions for analysis
CREATE TABLE IF NOT EXISTS Predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER NOT NULL,
    audio_file_id INTEGER NOT NULL,
    predicted_text TEXT NOT NULL,
    ground_truth TEXT,
    confidence_score REAL,
    inference_time_ms REAL,
    wer REAL,
    cer REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(training_run_id) REFERENCES TrainingRuns(id) ON DELETE CASCADE,
    FOREIGN KEY(audio_file_id) REFERENCES AudioFiles(id) ON DELETE CASCADE
);

-- Table: DataAugmentations
-- Track augmentation operations applied
CREATE TABLE IF NOT EXISTS DataAugmentations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER NOT NULL,
    augmentation_type TEXT NOT NULL,
    parameters TEXT,
    enabled INTEGER DEFAULT 1,
    FOREIGN KEY(training_run_id) REFERENCES TrainingRuns(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_audiofiles_language ON AudioFiles(language);
CREATE INDEX IF NOT EXISTS idx_audiofiles_speaker ON AudioFiles(speaker_id);
CREATE INDEX IF NOT EXISTS idx_datasplits_split ON DataSplits(split_type, split_version);
CREATE INDEX IF NOT EXISTS idx_trainingruns_status ON TrainingRuns(status);
CREATE INDEX IF NOT EXISTS idx_epochmetrics_run ON EpochMetrics(training_run_id, epoch);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON Predictions(training_run_id);

-- Views for easy querying
CREATE VIEW IF NOT EXISTS TrainingRunSummary AS
SELECT 
    tr.id,
    tr.run_name,
    tr.model_id,
    m.model_name,
    tr.batch_size,
    tr.learning_rate,
    tr.best_val_loss,
    tr.word_error_rate,
    tr.character_error_rate,
    tr.status,
    tr.started_at,
    tr.completed_at,
    tr.total_training_time_seconds
FROM TrainingRuns tr
LEFT JOIN Models m ON tr.model_id = m.id;

CREATE VIEW IF NOT EXISTS DatasetStats AS
SELECT 
    ds.split_type,
    ds.split_version,
    COUNT(DISTINCT af.id) as num_files,
    SUM(af.duration_seconds) as total_duration_seconds,
    AVG(af.duration_seconds) as avg_duration_seconds,
    COUNT(DISTINCT af.speaker_id) as num_speakers
FROM DataSplits ds
JOIN AudioFiles af ON ds.audio_file_id = af.id
GROUP BY ds.split_type, ds.split_version;

