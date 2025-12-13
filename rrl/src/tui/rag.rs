//! RAG (Retrieval-Augmented Generation) TUI
//!
//! Interactive interface for document Q&A with the RAG pipeline.

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::widgets::ListState;
use std::sync::mpsc::Sender;

// ============================================================================
// Constants
// ============================================================================

/// Available embedder models
pub const EMBEDDER_MODELS: &[(&str, &str)] = &[
    ("bert-base-uncased", "110M params, general purpose"),
    ("bert-large-uncased", "340M params, higher capacity"),
    ("distilbert-base-uncased", "66M params, faster inference"),
    ("Custom...", "Enter HuggingFace model ID"),
];

/// Available generator models
pub const GENERATOR_MODELS: &[(&str, &str)] = &[
    ("Qwen/Qwen2.5-0.5B", "0.5B params, fast generation"),
    ("Qwen/Qwen2.5-1.5B", "1.5B params, better quality"),
    ("Qwen/Qwen2.5-3B", "3B params, high quality"),
    ("Custom...", "Enter HuggingFace model ID"),
];

/// Available retrieval strategies
pub const RETRIEVAL_STRATEGIES: &[(&str, &str)] = &[
    ("hybrid", "HNSW + BM25 with RRF fusion (recommended)"),
    ("dense", "HNSW vector search only"),
    ("sparse", "BM25 keyword search only"),
];

/// Available hardware devices
pub const HARDWARE_DEVICES: &[(&str, &str, &str)] = &[
    ("auto", "Auto-detect", "Use best available (CUDA > Metal > CPU)"),
    ("cuda", "CUDA", "NVIDIA GPU acceleration"),
    ("metal", "Metal", "Apple Silicon GPU acceleration"),
    ("cpu", "CPU", "CPU only (slowest)"),
];

// ============================================================================
// Enums
// ============================================================================

/// RAG section for navigation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagSection {
    IndexPath,
    EmbedderModel,
    EmbedderCheckpoint,
    GeneratorModel,
    GeneratorCheckpoint,
    RetrievalStrategy,
    Hardware,
    QueryInput,
    Execute,
}

impl RagSection {
    pub fn all() -> &'static [RagSection] {
        &[
            RagSection::IndexPath,
            RagSection::EmbedderModel,
            RagSection::EmbedderCheckpoint,
            RagSection::GeneratorModel,
            RagSection::GeneratorCheckpoint,
            RagSection::RetrievalStrategy,
            RagSection::Hardware,
            RagSection::QueryInput,
            RagSection::Execute,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            RagSection::IndexPath => 0,
            RagSection::EmbedderModel => 1,
            RagSection::EmbedderCheckpoint => 2,
            RagSection::GeneratorModel => 3,
            RagSection::GeneratorCheckpoint => 4,
            RagSection::RetrievalStrategy => 5,
            RagSection::Hardware => 6,
            RagSection::QueryInput => 7,
            RagSection::Execute => 8,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index % 9 {
            0 => RagSection::IndexPath,
            1 => RagSection::EmbedderModel,
            2 => RagSection::EmbedderCheckpoint,
            3 => RagSection::GeneratorModel,
            4 => RagSection::GeneratorCheckpoint,
            5 => RagSection::RetrievalStrategy,
            6 => RagSection::Hardware,
            7 => RagSection::QueryInput,
            _ => RagSection::Execute,
        }
    }

    pub fn next(&self) -> Self {
        Self::from_index(self.index() + 1)
    }

    pub fn prev(&self) -> Self {
        Self::from_index((self.index() + 8) % 9)
    }

    pub fn name(&self) -> &'static str {
        match self {
            RagSection::IndexPath => "Index",
            RagSection::EmbedderModel => "Embedder",
            RagSection::EmbedderCheckpoint => "Emb Ckpt",
            RagSection::GeneratorModel => "Generator",
            RagSection::GeneratorCheckpoint => "Gen Ckpt",
            RagSection::RetrievalStrategy => "Strategy",
            RagSection::Hardware => "Hardware",
            RagSection::QueryInput => "Query",
            RagSection::Execute => "Execute",
        }
    }
}

/// RAG application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagState {
    Configuring,
    LoadingPipeline,
    Ready,
    Processing,
    Error,
}

// ============================================================================
// Data Structures
// ============================================================================

/// RAG query result for display
#[derive(Debug, Clone)]
pub struct RagDisplayResult {
    pub answer: String,
    pub sources: Vec<SourceDisplay>,
    pub retrieval_time_ms: u64,
    pub generation_time_ms: u64,
    pub total_time_ms: u64,
}

/// Source citation for display
#[derive(Debug, Clone)]
pub struct SourceDisplay {
    pub rank: usize,
    pub chunk_id: String,
    pub document_id: String,
    pub score: f32,
    pub snippet: String,
}

/// Messages for async pipeline operations
#[derive(Debug)]
pub enum RagMessage {
    LoadingProgress(String, f64),
    PipelineReady,
    QueryResult(RagDisplayResult),
    Error(String),
}

/// Query request for the pipeline thread
#[derive(Debug)]
pub struct QueryRequest {
    pub query: String,
    pub top_k: usize,
}

/// Main RAG tab state
pub struct RagApp {
    // Current state
    pub state: RagState,
    pub current_section: RagSection,

    // Index path
    pub index_path: String,
    pub editing_index_path: bool,

    // Embedder model selection
    pub embedder_list_state: ListState,
    pub selected_embedder_idx: usize,
    pub custom_embedder: String,
    pub editing_embedder: bool,

    // Embedder checkpoint (optional LoRA)
    pub embedder_checkpoint: String,
    pub editing_embedder_checkpoint: bool,

    // Generator model selection
    pub generator_list_state: ListState,
    pub selected_generator_idx: usize,
    pub custom_generator: String,
    pub editing_generator: bool,

    // Generator checkpoint (optional LoRA)
    pub generator_checkpoint: String,
    pub editing_generator_checkpoint: bool,

    // Retrieval strategy selection
    pub strategy_list_state: ListState,
    pub selected_strategy_idx: usize,

    // Hardware selection
    pub hardware_list_state: ListState,
    pub selected_hardware_idx: usize,

    // LoRA configuration
    pub lora_rank: usize,
    pub lora_alpha: f32,

    // Query input
    pub query_input: String,
    pub cursor_position: usize,

    // Results
    pub current_result: Option<RagDisplayResult>,
    pub query_history: Vec<(String, RagDisplayResult)>,

    // Control flags
    pub should_quit: bool,
    pub should_initialize: bool,
    pub should_execute_query: bool,

    // Status and errors
    pub status_message: String,
    pub error_message: Option<String>,
    pub loading_progress: f64,

    // Pipeline state (set after initialization)
    pub pipeline_ready: bool,

    // Query sender (set after pipeline init)
    pub query_sender: Option<std::sync::mpsc::Sender<QueryRequest>>,
}

impl Default for RagApp {
    fn default() -> Self {
        let mut embedder_list_state = ListState::default();
        embedder_list_state.select(Some(0));

        let mut generator_list_state = ListState::default();
        generator_list_state.select(Some(0));

        let mut strategy_list_state = ListState::default();
        strategy_list_state.select(Some(0));

        let mut hardware_list_state = ListState::default();
        hardware_list_state.select(Some(0));

        Self {
            state: RagState::Configuring,
            current_section: RagSection::IndexPath,

            index_path: "./output/indexes".to_string(),
            editing_index_path: false,

            embedder_list_state,
            selected_embedder_idx: 0,
            custom_embedder: String::new(),
            editing_embedder: false,

            embedder_checkpoint: String::new(),
            editing_embedder_checkpoint: false,

            generator_list_state,
            selected_generator_idx: 0,
            custom_generator: String::new(),
            editing_generator: false,

            generator_checkpoint: String::new(),
            editing_generator_checkpoint: false,

            strategy_list_state,
            selected_strategy_idx: 0,

            hardware_list_state,
            selected_hardware_idx: 0,

            lora_rank: 8,
            lora_alpha: 16.0,

            query_input: String::new(),
            cursor_position: 0,

            current_result: None,
            query_history: Vec::new(),

            should_quit: false,
            should_initialize: false,
            should_execute_query: false,

            status_message: "Configure RAG pipeline settings, then press Enter on Execute".to_string(),
            error_message: None,
            loading_progress: 0.0,

            pipeline_ready: false,

            query_sender: None,
        }
    }
}

impl RagApp {
    pub fn new() -> Self {
        Self::default()
    }

    // ========================================================================
    // Getters
    // ========================================================================

    pub fn get_embedder_model(&self) -> String {
        if self.selected_embedder_idx == EMBEDDER_MODELS.len() - 1 {
            self.custom_embedder.clone()
        } else {
            EMBEDDER_MODELS[self.selected_embedder_idx].0.to_string()
        }
    }

    pub fn get_generator_model(&self) -> String {
        if self.selected_generator_idx == GENERATOR_MODELS.len() - 1 {
            self.custom_generator.clone()
        } else {
            GENERATOR_MODELS[self.selected_generator_idx].0.to_string()
        }
    }

    pub fn get_retrieval_strategy(&self) -> String {
        RETRIEVAL_STRATEGIES[self.selected_strategy_idx].0.to_string()
    }

    pub fn get_device(&self) -> String {
        HARDWARE_DEVICES[self.selected_hardware_idx].0.to_string()
    }

    pub fn get_embedder_checkpoint(&self) -> Option<String> {
        if self.embedder_checkpoint.is_empty() {
            None
        } else {
            Some(self.embedder_checkpoint.clone())
        }
    }

    pub fn get_generator_checkpoint(&self) -> Option<String> {
        if self.generator_checkpoint.is_empty() {
            None
        } else {
            Some(self.generator_checkpoint.clone())
        }
    }

    // ========================================================================
    // Input Handling
    // ========================================================================

    pub fn is_editing(&self) -> bool {
        self.editing_index_path
            || self.editing_embedder
            || self.editing_embedder_checkpoint
            || self.editing_generator
            || self.editing_generator_checkpoint
    }

    pub fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        // Handle editing modes first
        if self.editing_index_path {
            self.handle_path_edit(key, EditTarget::IndexPath);
            return;
        }
        if self.editing_embedder {
            self.handle_path_edit(key, EditTarget::CustomEmbedder);
            return;
        }
        if self.editing_embedder_checkpoint {
            self.handle_path_edit(key, EditTarget::EmbedderCheckpoint);
            return;
        }
        if self.editing_generator {
            self.handle_path_edit(key, EditTarget::CustomGenerator);
            return;
        }
        if self.editing_generator_checkpoint {
            self.handle_path_edit(key, EditTarget::GeneratorCheckpoint);
            return;
        }

        // Handle query input mode
        if self.current_section == RagSection::QueryInput && self.state == RagState::Ready {
            self.handle_query_input(key, modifiers);
            return;
        }

        // Handle error state
        if self.state == RagState::Error {
            if key == KeyCode::Enter || key == KeyCode::Esc {
                self.clear_error();
            }
            return;
        }

        // Global navigation
        match key {
            KeyCode::Char('q') if modifiers.is_empty() => {
                self.should_quit = true;
            }
            KeyCode::Tab => {
                self.current_section = self.current_section.next();
            }
            KeyCode::BackTab => {
                self.current_section = self.current_section.prev();
            }
            _ => {
                // Section-specific handling
                match self.current_section {
                    RagSection::IndexPath => self.handle_index_path_key(key),
                    RagSection::EmbedderModel => self.handle_embedder_key(key),
                    RagSection::EmbedderCheckpoint => self.handle_embedder_checkpoint_key(key),
                    RagSection::GeneratorModel => self.handle_generator_key(key),
                    RagSection::GeneratorCheckpoint => self.handle_generator_checkpoint_key(key),
                    RagSection::RetrievalStrategy => self.handle_strategy_key(key),
                    RagSection::Hardware => self.handle_hardware_key(key),
                    RagSection::QueryInput => self.handle_query_section_key(key),
                    RagSection::Execute => self.handle_execute_key(key),
                }
            }
        }
    }

    fn handle_path_edit(&mut self, key: KeyCode, target: EditTarget) {
        match key {
            KeyCode::Enter | KeyCode::Esc => {
                match target {
                    EditTarget::IndexPath => self.editing_index_path = false,
                    EditTarget::CustomEmbedder => self.editing_embedder = false,
                    EditTarget::EmbedderCheckpoint => self.editing_embedder_checkpoint = false,
                    EditTarget::CustomGenerator => self.editing_generator = false,
                    EditTarget::GeneratorCheckpoint => self.editing_generator_checkpoint = false,
                }
            }
            KeyCode::Backspace => {
                let s = self.get_edit_string_mut(target);
                s.pop();
            }
            KeyCode::Char(c) => {
                let s = self.get_edit_string_mut(target);
                s.push(c);
            }
            _ => {}
        }
    }

    fn get_edit_string_mut(&mut self, target: EditTarget) -> &mut String {
        match target {
            EditTarget::IndexPath => &mut self.index_path,
            EditTarget::CustomEmbedder => &mut self.custom_embedder,
            EditTarget::EmbedderCheckpoint => &mut self.embedder_checkpoint,
            EditTarget::CustomGenerator => &mut self.custom_generator,
            EditTarget::GeneratorCheckpoint => &mut self.generator_checkpoint,
        }
    }

    fn handle_query_input(&mut self, key: KeyCode, _modifiers: KeyModifiers) {
        match key {
            KeyCode::Char(c) => {
                self.query_input.insert(self.cursor_position, c);
                self.cursor_position += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_position > 0 {
                    self.query_input.remove(self.cursor_position - 1);
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Delete => {
                if self.cursor_position < self.query_input.len() {
                    self.query_input.remove(self.cursor_position);
                }
            }
            KeyCode::Left => {
                if self.cursor_position > 0 {
                    self.cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if self.cursor_position < self.query_input.len() {
                    self.cursor_position += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_position = 0;
            }
            KeyCode::End => {
                self.cursor_position = self.query_input.len();
            }
            KeyCode::Enter => {
                if !self.query_input.is_empty() && self.pipeline_ready {
                    self.should_execute_query = true;
                }
            }
            KeyCode::Tab => {
                self.current_section = self.current_section.next();
            }
            KeyCode::Esc => {
                self.current_section = RagSection::Execute;
            }
            _ => {}
        }
    }

    fn handle_index_path_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            self.editing_index_path = true;
        }
    }

    fn handle_embedder_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_embedder_idx > 0 {
                    self.selected_embedder_idx -= 1;
                    self.embedder_list_state.select(Some(self.selected_embedder_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_embedder_idx < EMBEDDER_MODELS.len() - 1 {
                    self.selected_embedder_idx += 1;
                    self.embedder_list_state.select(Some(self.selected_embedder_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_embedder_idx == EMBEDDER_MODELS.len() - 1 {
                    self.editing_embedder = true;
                }
            }
            _ => {}
        }
    }

    fn handle_embedder_checkpoint_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            self.editing_embedder_checkpoint = true;
        }
    }

    fn handle_generator_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_generator_idx > 0 {
                    self.selected_generator_idx -= 1;
                    self.generator_list_state.select(Some(self.selected_generator_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_generator_idx < GENERATOR_MODELS.len() - 1 {
                    self.selected_generator_idx += 1;
                    self.generator_list_state.select(Some(self.selected_generator_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_generator_idx == GENERATOR_MODELS.len() - 1 {
                    self.editing_generator = true;
                }
            }
            _ => {}
        }
    }

    fn handle_generator_checkpoint_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            self.editing_generator_checkpoint = true;
        }
    }

    fn handle_strategy_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_strategy_idx > 0 {
                    self.selected_strategy_idx -= 1;
                    self.strategy_list_state.select(Some(self.selected_strategy_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_strategy_idx < RETRIEVAL_STRATEGIES.len() - 1 {
                    self.selected_strategy_idx += 1;
                    self.strategy_list_state.select(Some(self.selected_strategy_idx));
                }
            }
            _ => {}
        }
    }

    fn handle_hardware_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_hardware_idx > 0 {
                    self.selected_hardware_idx -= 1;
                    self.hardware_list_state.select(Some(self.selected_hardware_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_hardware_idx < HARDWARE_DEVICES.len() - 1 {
                    self.selected_hardware_idx += 1;
                    self.hardware_list_state.select(Some(self.selected_hardware_idx));
                }
            }
            _ => {}
        }
    }

    fn handle_query_section_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter && self.state == RagState::Ready {
            // Focus is already on query, start typing
        }
    }

    fn handle_execute_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            if self.validate_config() {
                if self.state == RagState::Configuring {
                    self.should_initialize = true;
                } else if self.state == RagState::Ready && !self.query_input.is_empty() {
                    self.should_execute_query = true;
                }
            }
        }
    }

    fn validate_config(&mut self) -> bool {
        // Validate index path
        if self.index_path.is_empty() {
            self.error_message = Some("Index path cannot be empty".to_string());
            return false;
        }

        // Validate embedder model
        let embedder = self.get_embedder_model();
        if embedder.is_empty() {
            self.error_message = Some("Embedder model cannot be empty".to_string());
            return false;
        }

        // Validate generator model
        let generator = self.get_generator_model();
        if generator.is_empty() {
            self.error_message = Some("Generator model cannot be empty".to_string());
            return false;
        }

        self.error_message = None;
        true
    }

    // ========================================================================
    // State Updates (called from main event loop)
    // ========================================================================

    pub fn set_loading(&mut self, message: &str, progress: f64) {
        self.state = RagState::LoadingPipeline;
        self.status_message = message.to_string();
        self.loading_progress = progress;
    }

    pub fn set_ready(&mut self) {
        self.state = RagState::Ready;
        self.pipeline_ready = true;
        self.status_message = "Pipeline ready. Enter a query and press Enter.".to_string();
        self.loading_progress = 1.0;
        self.current_section = RagSection::QueryInput;
    }

    pub fn set_processing(&mut self) {
        self.state = RagState::Processing;
        self.status_message = "Processing query...".to_string();
    }

    pub fn set_result(&mut self, result: RagDisplayResult) {
        // Save to history
        let query = self.query_input.clone();
        self.query_history.push((query, result.clone()));

        // Set current result
        self.current_result = Some(result);

        // Update state
        self.state = RagState::Ready;
        self.status_message = "Query completed. Enter another query or view results.".to_string();

        // Clear query input
        self.query_input.clear();
        self.cursor_position = 0;
    }

    pub fn set_error(&mut self, error: String) {
        self.state = RagState::Error;
        self.error_message = Some(error.clone());
        self.status_message = format!("Error: {}", error);
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
        if self.pipeline_ready {
            self.state = RagState::Ready;
        } else {
            self.state = RagState::Configuring;
        }
    }
}

// ============================================================================
// Helper Enums
// ============================================================================

#[derive(Clone, Copy)]
enum EditTarget {
    IndexPath,
    CustomEmbedder,
    EmbedderCheckpoint,
    CustomGenerator,
    GeneratorCheckpoint,
}

// ============================================================================
// Configuration Export
// ============================================================================

/// Configuration for initializing the RAG pipeline
#[derive(Debug, Clone)]
pub struct RagAppConfig {
    pub index_path: String,
    pub embedder_model: String,
    pub embedder_checkpoint: Option<String>,
    pub generator_model: String,
    pub generator_checkpoint: Option<String>,
    pub retrieval_strategy: String,
    pub device: String,
    pub lora_rank: usize,
    pub lora_alpha: f32,
}

impl From<&RagApp> for RagAppConfig {
    fn from(app: &RagApp) -> Self {
        Self {
            index_path: app.index_path.clone(),
            embedder_model: app.get_embedder_model(),
            embedder_checkpoint: app.get_embedder_checkpoint(),
            generator_model: app.get_generator_model(),
            generator_checkpoint: app.get_generator_checkpoint(),
            retrieval_strategy: app.get_retrieval_strategy(),
            device: app.get_device(),
            lora_rank: app.lora_rank,
            lora_alpha: app.lora_alpha,
        }
    }
}

// ============================================================================
// Pipeline Initialization (Background Thread)
// ============================================================================

use std::sync::mpsc::Receiver;

#[cfg(feature = "training")]
pub fn spawn_pipeline_init(
    config: RagAppConfig,
    tx: Sender<RagMessage>,
) -> std::sync::mpsc::Sender<QueryRequest> {
    use std::sync::mpsc::channel;
    use std::thread;

    // Create query channel
    let (query_tx, query_rx) = channel::<QueryRequest>();

    thread::spawn(move || {
        match initialize_pipeline_internal(&config, &tx) {
            Ok(pipeline) => {
                // Signal ready
                let _ = tx.send(RagMessage::PipelineReady);

                // Run query loop
                run_query_loop(pipeline, query_rx, tx);
            }
            Err(e) => {
                let _ = tx.send(RagMessage::Error(e.to_string()));
            }
        }
    });

    query_tx
}

#[cfg(feature = "training")]
fn initialize_pipeline_internal(
    config: &RagAppConfig,
    tx: &Sender<RagMessage>,
) -> anyhow::Result<crate::rag::RagPipeline> {
    use crate::embedding::backends::{CandleBertConfig, CandleBertEmbedder};
    use crate::embedding::Embedder;
    use crate::rag::{Generator, RagConfig, RagPipelineBuilder, RetrievalStrategy};
    use crate::rag::generation::GeneratorConfig;
    use crate::retrieval::{Bm25Retriever, HnswRetriever, HybridRetriever, Retriever};
    use crate::training::DevicePreference;
    use std::sync::Arc;

    // 1. Load embedder
    let _ = tx.send(RagMessage::LoadingProgress("Loading embedder model...".to_string(), 0.2));

    let device_pref: DevicePreference = config.device.parse()?;
    let mut embedder_config = CandleBertConfig::new(&config.embedder_model)
        .with_lora_config(config.lora_rank, config.lora_alpha)
        .with_device(device_pref.clone());

    if let Some(ref ckpt) = config.embedder_checkpoint {
        embedder_config = embedder_config.with_lora_checkpoint(ckpt);
    }

    let embedder: Arc<dyn Embedder> = Arc::new(CandleBertEmbedder::new(embedder_config)?);

    // 2. Load retriever
    let _ = tx.send(RagMessage::LoadingProgress("Loading retriever...".to_string(), 0.4));

    let index_path = std::path::Path::new(&config.index_path);
    let hnsw_dir = index_path.join("hnsw");
    let bm25_dir = index_path.join("bm25");

    let strategy = match config.retrieval_strategy.as_str() {
        "hybrid" => RetrievalStrategy::Hybrid,
        "dense" => RetrievalStrategy::Dense,
        "sparse" => RetrievalStrategy::Sparse,
        _ => RetrievalStrategy::Hybrid,
    };

    let retriever: Arc<dyn Retriever> = match config.retrieval_strategy.as_str() {
        "hybrid" => {
            if hnsw_dir.exists() && bm25_dir.exists() {
                let hnsw: Arc<dyn Retriever> = Arc::new(HnswRetriever::load(&hnsw_dir, embedder.clone())?);
                let bm25: Arc<dyn Retriever> = Arc::new(Bm25Retriever::load(&bm25_dir)?);
                Arc::new(HybridRetriever::new(vec![hnsw, bm25]))
            } else {
                return Err(anyhow::anyhow!("Hybrid strategy requires both HNSW and BM25 indexes"));
            }
        }
        "dense" => {
            if hnsw_dir.exists() {
                Arc::new(HnswRetriever::load(&hnsw_dir, embedder.clone())?)
            } else {
                return Err(anyhow::anyhow!("Dense strategy requires HNSW index"));
            }
        }
        "sparse" => {
            if bm25_dir.exists() {
                Arc::new(Bm25Retriever::load(&bm25_dir)?)
            } else {
                return Err(anyhow::anyhow!("Sparse strategy requires BM25 index"));
            }
        }
        _ => return Err(anyhow::anyhow!("Unknown retrieval strategy")),
    };

    // 3. Load generator
    let _ = tx.send(RagMessage::LoadingProgress("Loading generator model...".to_string(), 0.7));

    let generator_config = GeneratorConfig {
        model_id: config.generator_model.clone(),
        lora_checkpoint: config.generator_checkpoint.clone(),
        lora_rank: config.lora_rank,
        lora_alpha: config.lora_alpha,
        device: device_pref,
        ..Default::default()
    };

    let generator = Generator::new(generator_config)?;

    // 4. Build pipeline
    let _ = tx.send(RagMessage::LoadingProgress("Building pipeline...".to_string(), 0.9));

    let rag_config = RagConfig {
        retrieval_strategy: strategy,
        top_k: 5,
        max_context_chars: 4000,
        include_citations: true,
        ..Default::default()
    };

    let pipeline = RagPipelineBuilder::new()
        .embedder(embedder)
        .retriever(retriever)
        .generator(generator)
        .config(rag_config)
        .build()?;

    Ok(pipeline)
}

#[cfg(feature = "training")]
fn run_query_loop(
    pipeline: crate::rag::RagPipeline,
    query_rx: Receiver<QueryRequest>,
    tx: Sender<RagMessage>,
) {
    use crate::rag::RagQuery;
    use std::time::Instant;

    while let Ok(request) = query_rx.recv() {
        let start = Instant::now();

        // Create RagQuery with top_k
        let rag_query = RagQuery::new(&request.query).with_top_k(request.top_k);

        // Execute query
        match pipeline.query(rag_query) {
            Ok(response) => {
                let total_time = start.elapsed().as_millis() as u64;

                // Convert sources to display format
                let sources: Vec<SourceDisplay> = response
                    .sources
                    .iter()
                    .enumerate()
                    .map(|(i, src)| SourceDisplay {
                        rank: i + 1,
                        chunk_id: src.chunk_id.clone(),
                        document_id: src.document_id.clone(),
                        score: src.score,
                        snippet: src.snippet.clone(),
                    })
                    .collect();

                let result = RagDisplayResult {
                    answer: response.answer.clone(),
                    sources,
                    retrieval_time_ms: response.retrieval_time_ms,
                    generation_time_ms: response.generation_time_ms,
                    total_time_ms: total_time,
                };

                let _ = tx.send(RagMessage::QueryResult(result));
            }
            Err(e) => {
                let _ = tx.send(RagMessage::Error(format!("Query failed: {}", e)));
            }
        }
    }
}

#[cfg(not(feature = "training"))]
pub fn spawn_pipeline_init(
    _config: RagAppConfig,
    tx: Sender<RagMessage>,
) -> std::sync::mpsc::Sender<QueryRequest> {
    use std::sync::mpsc::channel;

    let (query_tx, _query_rx) = channel::<QueryRequest>();

    let _ = tx.send(RagMessage::Error(
        "Training feature not enabled. Compile with: cargo build --features training".to_string()
    ));

    query_tx
}

// ============================================================================
// Device Status Helper
// ============================================================================

pub fn get_device_status(device: &str) -> &'static str {
    match device {
        "auto" => "Available",
        "cpu" => "Available",
        "cuda" => {
            #[cfg(feature = "cuda")]
            { "Available" }
            #[cfg(not(feature = "cuda"))]
            { "Not Compiled" }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            { "Available" }
            #[cfg(not(feature = "metal"))]
            { "Not Compiled" }
        }
        _ => "Unknown",
    }
}
