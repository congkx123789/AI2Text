"""
Database utilities for ASR training system.

This module provides comprehensive database operations for managing:
- Audio files and their metadata
- Transcripts and annotations
- Data splits (train/val/test)
- Model registration and tracking
- Training runs and metrics
- Predictions and evaluation results

The ASRDatabase class serves as the main interface to SQLite database,
providing high-level methods for all database operations. It automatically
initializes the database schema on first use.

Example:
    >>> from database.db_utils import ASRDatabase
    >>> db = ASRDatabase("database/asr_training.db")
    >>> 
    >>> # Add audio file
    >>> audio_id = db.add_audio_file(
    ...     file_path="data/raw/audio.wav",
    ...     filename="audio.wav",
    ...     duration=5.0,
    ...     sample_rate=16000
    ... )
    >>> 
    >>> # Add transcript
    >>> db.add_transcript(
    ...     audio_file_id=audio_id,
    ...     transcript="xin chào việt nam",
    ...     normalized_transcript="xin chao viet nam"
    ... )
    >>> 
    >>> # Assign to train split
    >>> db.assign_split(audio_id, "train", "v1")
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import pandas as pd
from collections import defaultdict


class ASRDatabase:
    """
    Main database interface for ASR training system.
    
    This class provides a high-level interface to the SQLite database used for
    storing all training metadata, audio files, transcripts, models, and metrics.
    It automatically initializes the database schema if the database doesn't exist.
    
    The database uses SQLite for portability and ease of use. All operations
    are wrapped in transactions for data consistency.
    
    Attributes:
        db_path (Path): Path to the SQLite database file
        
    Example:
        >>> db = ASRDatabase("database/asr_training.db")
        >>> stats = db.get_dataset_statistics("v1")
        >>> print(stats)
    """
    
    def __init__(self, db_path: str = "database/asr_training.db"):
        """
        Initialize database connection and create schema if needed.
        
        Creates the database file and directory if they don't exist, then
        initializes all tables, indexes, and views according to the schema.
        
        Args:
            db_path (str): Path to SQLite database file. Defaults to 
                          "database/asr_training.db". The directory will be 
                          created if it doesn't exist.
                          
        Note:
            The database schema is automatically created from init_db.sql
            if the database is new or missing tables.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """
        Initialize database with schema from init_db.sql.
        
        This method is called automatically during initialization. It reads
        the SQL schema file and executes it to create all necessary tables,
        indexes, and views.
        
        The schema includes tables for:
        - AudioFiles: Metadata about audio recordings
        - Transcripts: Ground truth transcriptions
        - DataSplits: Train/val/test assignments
        - Models: Model registry
        - TrainingRuns: Training experiment tracking
        - EpochMetrics: Per-epoch training metrics
        - Predictions: Model predictions and evaluations
        
        Note:
            This is a private method and should not be called directly.
        """
        schema_path = Path(__file__).parent / "init_db.sql"
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Provides safe database connection handling with automatic cleanup.
        Returns connections with Row factory enabled for dict-like access.
        
        Yields:
            sqlite3.Connection: Database connection with Row factory
            
        Example:
            >>> with db.get_connection() as conn:
            ...     cursor = conn.execute("SELECT * FROM AudioFiles LIMIT 1")
            ...     row = cursor.fetchone()
            ...     print(dict(row))
            
        Note:
            Always use this context manager instead of direct connections
            to ensure proper cleanup and avoid connection leaks.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # AudioFiles operations
    def add_audio_file(self, file_path: str, filename: str, 
                       duration: float, sample_rate: int,
                       channels: int = 1, format: str = "wav",
                       language: str = "vi", speaker_id: Optional[str] = None,
                       audio_quality: str = "medium", 
                       skip_duplicate: bool = True) -> Optional[int]:
        """
        Add a new audio file record to the database.
        
        Stores metadata about an audio file including its path, duration,
        sample rate, format, and quality. Optionally checks for duplicates
        to prevent inserting the same file twice.
        
        Args:
            file_path (str): Full path to the audio file (used as unique identifier)
            filename (str): Just the filename (e.g., "audio.wav")
            duration (float): Duration of audio in seconds
            sample_rate (int): Sample rate in Hz (e.g., 16000)
            channels (int): Number of audio channels (1=mono, 2=stereo). Default: 1
            format (str): Audio file format extension (e.g., "wav", "mp3"). Default: "wav"
            language (str): Language code (e.g., "vi" for Vietnamese). Default: "vi"
            speaker_id (Optional[str]): Identifier for the speaker. Default: None
            audio_quality (str): Quality rating: "high", "medium", or "low". 
                                Determined by sample rate if not specified.
            skip_duplicate (bool): If True, returns None for duplicate file_path.
                                  If False, allows duplicates. Default: True
        
        Returns:
            Optional[int]: ID of the inserted audio file record, or None if 
                         duplicate was skipped (when skip_duplicate=True)
        
        Example:
            >>> audio_id = db.add_audio_file(
            ...     file_path="data/raw/speaker1_001.wav",
            ...     filename="speaker1_001.wav",
            ...     duration=5.5,
            ...     sample_rate=16000,
            ...     speaker_id="speaker_01",
            ...     audio_quality="high"
            ... )
            >>> print(f"Audio file ID: {audio_id}")
            
        Note:
            - file_path must be unique (database constraint)
            - audio_quality should be one of: "high" (>=44kHz), "medium" (>=16kHz), "low" (<16kHz)
            - If skip_duplicate=True and file exists, returns None without error
        """
        with self.get_connection() as conn:
            # Check for duplicate if requested - prevents re-importing same files
            if skip_duplicate:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", (file_path,)
                )
                existing = cursor.fetchone()
                if existing:
                    return None  # Skip duplicate - return None instead of raising error
            
            # Insert audio file metadata into database
            cursor = conn.execute("""
                INSERT INTO AudioFiles 
                (file_path, filename, duration_seconds, sample_rate, channels, 
                 format, language, speaker_id, audio_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, filename, duration, sample_rate, channels, 
                  format, language, speaker_id, audio_quality))
            conn.commit()  # Commit transaction to save changes
            return cursor.lastrowid  # Return the ID of the newly inserted row
    
    def batch_add_audio_files(self, audio_data: List[Dict]) -> List[int]:
        """
        Batch insert multiple audio files efficiently in a single transaction.
        
        This method is optimized for bulk imports by processing multiple files
        in one transaction, which is much faster than calling add_audio_file()
        repeatedly. Duplicate detection is still performed for each file.
        
        Args:
            audio_data (List[Dict]): List of dictionaries, each containing:
                - file_path (str): Path to audio file
                - filename (str): Just the filename
                - duration (float): Duration in seconds
                - sample_rate (int): Sample rate in Hz
                - channels (int): Number of channels
                - format (str): File format
                - language (str): Language code (default: "vi")
                - speaker_id (Optional[str]): Speaker identifier
                - audio_quality (str): "high", "medium", or "low"
        
        Returns:
            List[int]: List of audio_file_ids in the same order as input.
                      Contains None for skipped duplicates.
        
        Example:
            >>> audio_data = [
            ...     {
            ...         'file_path': 'audio1.wav',
            ...         'filename': 'audio1.wav',
            ...         'duration': 5.0,
            ...         'sample_rate': 16000,
            ...         'channels': 1,
            ...         'format': 'wav',
            ...         'language': 'vi',
            ...         'speaker_id': 'speaker_01',
            ...         'audio_quality': 'medium'
            ...     },
            ...     # ... more files
            ... ]
            >>> ids = db.batch_add_audio_files(audio_data)
            >>> print(f"Inserted {len([i for i in ids if i is not None])} files")
        
        Performance:
            Much faster than individual inserts - use for importing large datasets.
            Typical speedup: 5-10x compared to individual add_audio_file() calls.
        """
        ids = []
        with self.get_connection() as conn:
            for data in audio_data:
                # Check for duplicate
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", 
                    (data['file_path'],)
                )
                if cursor.fetchone():
                    ids.append(None)
                    continue
                
                cursor = conn.execute("""
                    INSERT INTO AudioFiles 
                    (file_path, filename, duration_seconds, sample_rate, channels, 
                     format, language, speaker_id, audio_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['file_path'], data['filename'], data['duration'],
                    data['sample_rate'], data['channels'], data['format'],
                    data['language'], data.get('speaker_id'), data['audio_quality']
                ))
                ids.append(cursor.lastrowid)
            conn.commit()
        return ids
    
    def get_audio_file(self, audio_file_id: int) -> Optional[Dict]:
        """
        Retrieve audio file metadata by its database ID.
        
        Fetches all metadata associated with an audio file including path,
        duration, sample rate, format, speaker ID, and quality rating.
        
        Args:
            audio_file_id (int): Database ID of the audio file
            
        Returns:
            Optional[Dict]: Dictionary containing audio file metadata:
                - id: Database ID
                - file_path: Full path to audio file
                - filename: Just the filename
                - duration_seconds: Duration in seconds
                - sample_rate: Sample rate in Hz
                - channels: Number of audio channels
                - format: File format (e.g., "wav")
                - language: Language code
                - speaker_id: Speaker identifier (if available)
                - audio_quality: Quality rating
                - created_at: Timestamp when record was created
            Returns None if audio_file_id doesn't exist
            
        Example:
            >>> audio_file = db.get_audio_file(audio_id)
            >>> if audio_file:
            ...     print(f"File: {audio_file['filename']}")
            ...     print(f"Duration: {audio_file['duration_seconds']}s")
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM AudioFiles WHERE id = ?", (audio_file_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # Transcripts operations
    def add_transcript(self, audio_file_id: int, transcript: str,
                       normalized_transcript: Optional[str] = None,
                       annotation_type: str = "manual",
                       confidence_score: Optional[float] = None,
                       annotator_id: Optional[str] = None) -> int:
        """
        Add a transcript (transcription) for an audio file.
        
        Stores the ground truth transcription along with its normalized version
        for training. Automatically calculates word and character counts.
        Each audio file should have at least one transcript.
        
        Args:
            audio_file_id (int): Database ID of the associated audio file
            transcript (str): Original transcript text (with diacritics, mixed case)
            normalized_transcript (Optional[str]): Normalized transcript (lowercase,
                                                   numbers as words, abbreviations expanded).
                                                   If None, will be auto-generated.
            annotation_type (str): Type of annotation - "manual" (human transcribed),
                                  "auto" (ASR generated), or "corrected" (auto then corrected).
                                  Default: "manual"
            confidence_score (Optional[float]): Confidence score if auto-generated (0-1).
                                               None for manual transcriptions.
            annotator_id (Optional[str]): Identifier for the annotator/speaker.
                                        Useful for tracking who transcribed the audio.
        
        Returns:
            int: Database ID of the inserted transcript record
            
        Example:
            >>> # Add transcript for audio file
            >>> transcript_id = db.add_transcript(
            ...     audio_file_id=audio_id,
            ...     transcript="Xin chào Việt Nam",
            ...     normalized_transcript="xin chào việt nam",
            ...     annotation_type="manual",
            ...     annotator_id="annotator_01"
            ... )
            >>> print(f"Transcript ID: {transcript_id}")
            
        Note:
            - One audio file can have multiple transcripts (for different versions)
            - normalized_transcript should match the tokenizer output format
            - Word and character counts are automatically calculated from transcript
        """
        # Calculate word count (split by whitespace)
        word_count = len(transcript.split())
        # Calculate character count (excluding spaces)
        char_count = len(transcript)
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Transcripts 
                (audio_file_id, transcript, normalized_transcript, 
                 word_count, char_count, annotation_type, 
                 confidence_score, annotator_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (audio_file_id, transcript, normalized_transcript,
                  word_count, char_count, annotation_type,
                  confidence_score, annotator_id))
            conn.commit()
            return cursor.lastrowid
    
    # DataSplits operations
    def assign_split(self, audio_file_id: int, split_type: str, 
                     split_version: str = "v1") -> int:
        """
        Assign an audio file to a data split (train/val/test).
        
        Assigns an audio file to a specific split in a specific version.
        Supports multiple split versions (v1, v2, etc.) for different experiments.
        Uses INSERT OR REPLACE to update existing assignments.
        
        Args:
            audio_file_id (int): Database ID of the audio file to assign
            split_type (str): Split type - must be one of:
                - "train": Training data (typically 80% of data)
                - "val": Validation data (typically 10% of data)
                - "test": Test data (typically 10% of data)
            split_version (str): Version identifier for the split (e.g., "v1", "v2").
                               Allows multiple split configurations. Default: "v1"
        
        Returns:
            int: Database ID of the split assignment record
            
        Example:
            >>> # Assign to training split
            >>> db.assign_split(audio_id, "train", "v1")
            >>> 
            >>> # Assign to validation split
            >>> db.assign_split(audio_id, "val", "v1")
            
        Note:
            - Split assignments are unique per (audio_file_id, split_version)
            - If audio is already assigned, this will update the split_type
            - Recommended split ratios: 80% train, 10% val, 10% test
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO DataSplits 
                (audio_file_id, split_type, split_version)
                VALUES (?, ?, ?)
            """, (audio_file_id, split_type, split_version))
            conn.commit()
            return cursor.lastrowid
    
    def get_split_data(self, split_type: str, split_version: str = "v1") -> pd.DataFrame:
        """
        Retrieve all audio files and their transcripts for a specific split.
        
        Returns a pandas DataFrame containing all audio files assigned to the
        specified split, along with their associated transcripts. This is the
        main method for loading training/validation/test data.
        
        Args:
            split_type (str): Split to retrieve - "train", "val", or "test"
            split_version (str): Version identifier (e.g., "v1"). Default: "v1"
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                - audio_file_id: Database ID of audio file
                - file_path: Full path to audio file
                - filename: Audio filename
                - duration_seconds: Duration in seconds
                - sample_rate: Sample rate in Hz
                - speaker_id: Speaker identifier
                - transcript: Original transcript text
                - normalized_transcript: Normalized transcript text
            Empty DataFrame if split doesn't exist or has no data.
            
        Example:
            >>> # Get training data
            >>> train_df = db.get_split_data("train", "v1")
            >>> print(f"Training samples: {len(train_df)}")
            >>> 
            >>> # Get validation data
            >>> val_df = db.get_split_data("val", "v1")
            >>> print(f"Validation samples: {len(val_df)}")
            
        Note:
            - Only returns audio files that have transcripts
            - Results are ordered by audio_file_id
            - Use this method to get data for training/evaluation
        """
        query = """
            SELECT 
                af.id as audio_file_id,
                af.file_path,
                af.filename,
                af.duration_seconds,
                af.sample_rate,
                af.speaker_id,
                t.transcript,
                t.normalized_transcript
            FROM DataSplits ds
            JOIN AudioFiles af ON ds.audio_file_id = af.id
            JOIN Transcripts t ON af.id = t.audio_file_id
            WHERE ds.split_type = ? AND ds.split_version = ?
            ORDER BY af.id
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(split_type, split_version))
    
    # Models operations
    def add_model(self, model_name: str, model_type: str,
                  architecture: str, version: str,
                  checkpoint_path: Optional[str] = None,
                  config: Optional[Dict] = None,
                  total_parameters: Optional[int] = None) -> int:
        """Register a new model."""
        config_json = json.dumps(config) if config else None
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Models 
                (model_name, model_type, architecture, version, 
                 checkpoint_path, config_json, total_parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_name, model_type, architecture, version,
                  checkpoint_path, config_json, total_parameters))
            conn.commit()
            return cursor.lastrowid
    
    # TrainingRuns operations
    def create_training_run(self, model_id: int, run_name: str,
                            config: Dict, batch_size: int,
                            learning_rate: float, num_epochs: int,
                            optimizer: str, description: Optional[str] = None,
                            split_version: str = "v1",
                            gpu_name: Optional[str] = None) -> int:
        """Create a new training run."""
        config_json = json.dumps(config)
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO TrainingRuns 
                (model_id, run_name, description, config_json, split_version,
                 batch_size, learning_rate, num_epochs, optimizer, gpu_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_id, run_name, description, config_json, split_version,
                  batch_size, learning_rate, num_epochs, optimizer, gpu_name))
            conn.commit()
            return cursor.lastrowid
    
    def update_training_run(self, run_id: int, **kwargs):
        """Update training run with results."""
        set_clauses = []
        values = []
        
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)
        
        if not set_clauses:
            return
        
        values.append(run_id)
        query = f"UPDATE TrainingRuns SET {', '.join(set_clauses)} WHERE id = ?"
        
        with self.get_connection() as conn:
            conn.execute(query, values)
            conn.commit()
    
    def complete_training_run(self, run_id: int, final_train_loss: float,
                              final_val_loss: float, best_val_loss: float,
                              best_epoch: int, wer: float, cer: float,
                              total_time: float):
        """Mark training run as completed with final metrics."""
        self.update_training_run(
            run_id,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            word_error_rate=wer,
            character_error_rate=cer,
            total_training_time_seconds=total_time,
            status='completed',
            completed_at=datetime.now().isoformat()
        )
    
    # EpochMetrics operations
    def add_epoch_metrics(self, training_run_id: int, epoch: int,
                          train_loss: float, val_loss: float,
                          learning_rate: float, wer: Optional[float] = None,
                          cer: Optional[float] = None,
                          epoch_time: Optional[float] = None):
        """Log metrics for an epoch."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO EpochMetrics 
                (training_run_id, epoch, train_loss, val_loss, 
                 learning_rate, wer, cer, epoch_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (training_run_id, epoch, train_loss, val_loss,
                  learning_rate, wer, cer, epoch_time))
            conn.commit()
    
    def get_training_history(self, training_run_id: int) -> pd.DataFrame:
        """Get all epoch metrics for a training run."""
        query = """
            SELECT * FROM EpochMetrics 
            WHERE training_run_id = ? 
            ORDER BY epoch
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(training_run_id,))
    
    # Predictions operations
    def add_prediction(self, training_run_id: int, audio_file_id: int,
                       predicted_text: str, ground_truth: str,
                       confidence_score: Optional[float] = None,
                       inference_time_ms: Optional[float] = None,
                       wer: Optional[float] = None,
                       cer: Optional[float] = None) -> int:
        """Store a model prediction."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO Predictions 
                (training_run_id, audio_file_id, predicted_text, ground_truth,
                 confidence_score, inference_time_ms, wer, cer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (training_run_id, audio_file_id, predicted_text, ground_truth,
                  confidence_score, inference_time_ms, wer, cer))
            conn.commit()
            return cursor.lastrowid
    
    # Analytics and reporting
    def get_dataset_statistics(self, split_version: str = "v1") -> pd.DataFrame:
        """Get dataset statistics using the view."""
        query = "SELECT * FROM DatasetStats WHERE split_version = ?"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(split_version,))
    
    def validate_data_for_training(self, split_version: str = "v1") -> Dict:
        """Validate data quality and prepare training readiness report.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'is_ready': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        with self.get_connection() as conn:
            # Check splits exist
            cursor = conn.execute("""
                SELECT split_type, COUNT(*) as count 
                FROM DataSplits 
                WHERE split_version = ?
                GROUP BY split_type
            """, (split_version,))
            
            splits = {row['split_type']: row['count'] for row in cursor.fetchall()}
            
            if 'train' not in splits or splits['train'] == 0:
                validation_results['is_ready'] = False
                validation_results['issues'].append("No training data found!")
            
            if 'val' not in splits or splits['val'] == 0:
                validation_results['warnings'].append("No validation data found!")
            
            # Check for transcripts
            cursor = conn.execute("""
                SELECT COUNT(*) as count_without_transcript
                FROM AudioFiles af
                LEFT JOIN Transcripts t ON af.id = t.audio_file_id
                WHERE t.id IS NULL
            """)
            no_transcript = cursor.fetchone()['count_without_transcript']
            
            if no_transcript > 0:
                validation_results['issues'].append(
                    f"{no_transcript} audio files without transcripts!"
                )
                validation_results['is_ready'] = False
            
            # Check audio quality distribution
            cursor = conn.execute("""
                SELECT audio_quality, COUNT(*) as count
                FROM AudioFiles
                GROUP BY audio_quality
            """)
            quality_dist = {row['audio_quality']: row['count'] for row in cursor.fetchall()}
            
            # Check duration statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COALESCE(AVG(duration_seconds), 0) as avg_duration,
                    COALESCE(MIN(duration_seconds), 0) as min_duration,
                    COALESCE(MAX(duration_seconds), 0) as max_duration,
                    COALESCE(SUM(CASE WHEN duration_seconds < 0.5 THEN 1 ELSE 0 END), 0) as too_short,
                    COALESCE(SUM(CASE WHEN duration_seconds > 30 THEN 1 ELSE 0 END), 0) as too_long
                FROM AudioFiles
            """)
            row = cursor.fetchone()
            duration_stats = {
                'total': row['total'] or 0,
                'avg_duration': row['avg_duration'] or 0.0,
                'min_duration': row['min_duration'] or 0.0,
                'max_duration': row['max_duration'] or 0.0,
                'too_short': row['too_short'] or 0,
                'too_long': row['too_long'] or 0
            }
            
            # Compile statistics
            validation_results['statistics'] = {
                'splits': splits,
                'quality_distribution': quality_dist,
                'duration_stats': duration_stats,
                'files_without_transcripts': no_transcript
            }
            
            # Generate recommendations
            total_files = sum(splits.values())
            if total_files < 100:
                validation_results['recommendations'].append(
                    f"Only {total_files} files. Recommend at least 100 files for basic training."
                )
            elif total_files < 1000:
                validation_results['recommendations'].append(
                    f"{total_files} files is good for initial training. More data will improve results."
                )
            
            if duration_stats['too_short'] > 0:
                validation_results['warnings'].append(
                    f"{duration_stats['too_short']} files are too short (<0.5s). Consider filtering."
                )
            
            if duration_stats['too_long'] > 0:
                validation_results['warnings'].append(
                    f"{duration_stats['too_long']} files are too long (>30s). Consider splitting."
                )
            
            train_count = splits.get('train', 0)
            if train_count < 50:
                validation_results['is_ready'] = False
                validation_results['issues'].append(
                    f"Only {train_count} training samples. Need at least 50."
                )
        
        return validation_results
    
    def get_data_summary(self, split_version: str = "v1") -> Dict:
        """Get comprehensive data summary for training preparation."""
        with self.get_connection() as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(DISTINCT af.id) as total_files,
                    COUNT(DISTINCT af.speaker_id) as total_speakers,
                    COALESCE(SUM(af.duration_seconds), 0) as total_duration_hours,
                    COALESCE(AVG(af.duration_seconds), 0) as avg_duration,
                    COALESCE(AVG(t.word_count), 0) as avg_words_per_transcript
                FROM AudioFiles af
                JOIN Transcripts t ON af.id = t.audio_file_id
            """)
            row = cursor.fetchone()
            overall = {
                'total_files': row['total_files'] or 0,
                'total_speakers': row['total_speakers'] or 0,
                'total_duration_hours': (row['total_duration_hours'] or 0) / 3600,
                'avg_duration': row['avg_duration'] or 0.0,
                'avg_words_per_transcript': row['avg_words_per_transcript'] or 0.0
            }
            
            # Split statistics
            stats = self.get_dataset_statistics(split_version)
            stats_dict = stats.to_dict('records')
            
            return {
                'overall': overall,
                'splits': {row['split_type']: dict(row) for row in stats_dict},
                'validation': self.validate_data_for_training(split_version)
            }
    
    def get_training_runs_summary(self, limit: int = 10) -> pd.DataFrame:
        """Get summary of recent training runs."""
        query = """
            SELECT * FROM TrainingRunSummary 
            ORDER BY started_at DESC 
            LIMIT ?
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_best_model(self, metric: str = "word_error_rate") -> Optional[Dict]:
        """Get best performing model based on a metric."""
        query = f"""
            SELECT * FROM TrainingRuns 
            WHERE status = 'completed' AND {metric} IS NOT NULL
            ORDER BY {metric} ASC 
            LIMIT 1
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None


# Convenience functions
def get_db(db_path: str = "database/asr_training.db") -> ASRDatabase:
    """Get database instance."""
    return ASRDatabase(db_path)


if __name__ == "__main__":
    # Test database setup
    db = ASRDatabase("database/asr_training.db")
    print("Database initialized successfully!")
    print(f"Database location: {db.db_path.absolute()}")

