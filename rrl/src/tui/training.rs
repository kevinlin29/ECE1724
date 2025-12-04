//! Complete Training & Evaluation TUI
//!
//! A comprehensive terminal-based UI for model fine-tuning and evaluation.
//! Supports model selection, dataset configuration, hardware acceleration,
//! training parameters, and results display.

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap},
    Frame, Terminal,
};
use std::io;
use std::time::Duration;

// ============================================================================
// Constants
// ============================================================================

/// Available pre-defined models
pub const AVAILABLE_MODELS: &[(&str, &str)] = &[
    ("bert-base-uncased", "110M params, general purpose"),
    ("bert-large-uncased", "340M params, higher capacity"),
    ("distilbert-base-uncased", "66M params, faster training"),
    ("roberta-base", "125M params, robust pretraining"),
    ("Custom...", "Enter HuggingFace model ID"),
];

/// Available pre-defined training datasets
pub const TRAIN_DATASETS: &[(&str, &str)] = &[
    ("recipe-mpr/train.json", "Recipe-MPR training (400 examples)"),
    ("Custom...", "Enter custom path"),
];

/// Available pre-defined validation datasets
pub const VAL_DATASETS: &[(&str, &str)] = &[
    ("None", "Skip validation during training"),
    ("recipe-mpr/val.json", "Recipe-MPR validation"),
    ("Custom...", "Enter custom path"),
];

/// Available pre-defined test datasets
pub const TEST_DATASETS: &[(&str, &str)] = &[
    ("recipe-mpr/test.json", "Recipe-MPR test (50 examples)"),
    ("recipe-mpr/val.json", "Recipe-MPR validation"),
    ("Custom...", "Enter custom path"),
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

/// UI Section/Tab
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Model,
    TrainData,
    ValData,
    TestData,
    Hardware,
    Params,
    Output,
    Start,
}

impl Section {
    pub fn all() -> &'static [Section] {
        &[
            Section::Model,
            Section::TrainData,
            Section::ValData,
            Section::TestData,
            Section::Hardware,
            Section::Params,
            Section::Output,
            Section::Start,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            Section::Model => 0,
            Section::TrainData => 1,
            Section::ValData => 2,
            Section::TestData => 3,
            Section::Hardware => 4,
            Section::Params => 5,
            Section::Output => 6,
            Section::Start => 7,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index % 8 {
            0 => Section::Model,
            1 => Section::TrainData,
            2 => Section::ValData,
            3 => Section::TestData,
            4 => Section::Hardware,
            5 => Section::Params,
            6 => Section::Output,
            _ => Section::Start,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Section::Model => "Model",
            Section::TrainData => "Train",
            Section::ValData => "Val",
            Section::TestData => "Test",
            Section::Hardware => "Hardware",
            Section::Params => "Params",
            Section::Output => "Output",
            Section::Start => "Start",
        }
    }

    pub fn next(&self) -> Self {
        Self::from_index(self.index() + 1)
    }

    pub fn prev(&self) -> Self {
        Self::from_index((self.index() + 7) % 8)
    }
}

/// Parameter fields that can be edited
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamField {
    Epochs,
    LearningRate,
    BatchSize,
    LoraRank,
    LoraAlpha,
    MaxSeqLength,
    GradientAccum,
    WarmupRatio,
    SaveSteps,
    LoggingSteps,
}

impl ParamField {
    pub fn all() -> &'static [ParamField] {
        &[
            ParamField::Epochs,
            ParamField::LearningRate,
            ParamField::BatchSize,
            ParamField::LoraRank,
            ParamField::LoraAlpha,
            ParamField::MaxSeqLength,
            ParamField::GradientAccum,
            ParamField::WarmupRatio,
            ParamField::SaveSteps,
            ParamField::LoggingSteps,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ParamField::Epochs => "Epochs",
            ParamField::LearningRate => "Learning Rate",
            ParamField::BatchSize => "Batch Size",
            ParamField::LoraRank => "LoRA Rank",
            ParamField::LoraAlpha => "LoRA Alpha",
            ParamField::MaxSeqLength => "Max Seq Len",
            ParamField::GradientAccum => "Grad Accum",
            ParamField::WarmupRatio => "Warmup Ratio",
            ParamField::SaveSteps => "Save Steps",
            ParamField::LoggingSteps => "Log Steps",
        }
    }
}

/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Configuring,
    Training,
    Evaluating,
    Results,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Training progress information
#[derive(Debug, Clone, Default)]
pub struct TrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub step: usize,
    pub total_steps: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub samples_per_sec: f64,
    pub status_message: String,
}

/// Training results after completion
#[derive(Debug, Clone, Default)]
pub struct TrainingResults {
    pub success: bool,
    pub checkpoint_path: String,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub training_time_secs: f64,
    pub test_accuracy: f64,
    pub test_correct: usize,
    pub test_total: usize,
    pub base_accuracy: Option<f64>,
    pub error_message: Option<String>,
}

/// Main application state
pub struct TrainingApp {
    // Current state
    pub state: AppState,
    pub current_section: Section,

    // Model selection
    pub model_list_state: ListState,
    pub selected_model_idx: usize,
    pub custom_model: String,
    pub editing_model: bool,

    // Training data selection
    pub train_data_list_state: ListState,
    pub selected_train_idx: usize,
    pub custom_train_data: String,
    pub editing_train_data: bool,

    // Validation data selection
    pub val_data_list_state: ListState,
    pub selected_val_idx: usize,
    pub custom_val_data: String,
    pub editing_val_data: bool,

    // Test data selection
    pub test_data_list_state: ListState,
    pub selected_test_idx: usize,
    pub custom_test_data: String,
    pub editing_test_data: bool,

    // Hardware selection
    pub hardware_list_state: ListState,
    pub selected_hardware_idx: usize,

    // Training parameters
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub max_seq_length: usize,
    pub gradient_accum: usize,
    pub warmup_ratio: f64,
    pub save_steps: usize,
    pub logging_steps: usize,

    // Parameter editing
    pub selected_param_idx: usize,
    pub editing_param: bool,
    pub param_input: String,

    // Output directory
    pub output_dir: String,
    pub editing_output: bool,

    // Control flags
    pub should_quit: bool,
    pub should_start: bool,

    // Progress and results
    pub progress: TrainingProgress,
    pub results: TrainingResults,
    pub error_message: Option<String>,
}

impl Default for TrainingApp {
    fn default() -> Self {
        let mut model_list_state = ListState::default();
        model_list_state.select(Some(0));

        let mut train_data_list_state = ListState::default();
        train_data_list_state.select(Some(0));

        let mut val_data_list_state = ListState::default();
        val_data_list_state.select(Some(0));

        let mut test_data_list_state = ListState::default();
        test_data_list_state.select(Some(0));

        let mut hardware_list_state = ListState::default();
        hardware_list_state.select(Some(0));

        Self {
            state: AppState::Configuring,
            current_section: Section::Model,

            model_list_state,
            selected_model_idx: 0,
            custom_model: String::new(),
            editing_model: false,

            train_data_list_state,
            selected_train_idx: 0,
            custom_train_data: String::new(),
            editing_train_data: false,

            val_data_list_state,
            selected_val_idx: 0,
            custom_val_data: String::new(),
            editing_val_data: false,

            test_data_list_state,
            selected_test_idx: 0,
            custom_test_data: String::new(),
            editing_test_data: false,

            hardware_list_state,
            selected_hardware_idx: 0,

            epochs: 3,
            learning_rate: 5e-5,
            batch_size: 32,
            lora_rank: 8,
            lora_alpha: 16.0,
            max_seq_length: 512,
            gradient_accum: 1,
            warmup_ratio: 0.1,
            save_steps: 500,
            logging_steps: 100,

            selected_param_idx: 0,
            editing_param: false,
            param_input: String::new(),

            output_dir: "./output/model-lora".to_string(),
            editing_output: false,

            should_quit: false,
            should_start: false,

            progress: TrainingProgress::default(),
            results: TrainingResults::default(),
            error_message: None,
        }
    }
}

impl TrainingApp {
    pub fn new() -> Self {
        Self::default()
    }

    // ========================================================================
    // Getters
    // ========================================================================

    pub fn get_model(&self) -> String {
        if self.selected_model_idx == AVAILABLE_MODELS.len() - 1 {
            self.custom_model.clone()
        } else {
            AVAILABLE_MODELS[self.selected_model_idx].0.to_string()
        }
    }

    pub fn get_train_data(&self) -> String {
        if self.selected_train_idx == TRAIN_DATASETS.len() - 1 {
            self.custom_train_data.clone()
        } else {
            TRAIN_DATASETS[self.selected_train_idx].0.to_string()
        }
    }

    pub fn get_val_data(&self) -> Option<String> {
        if self.selected_val_idx == 0 {
            None // "None" selected
        } else if self.selected_val_idx == VAL_DATASETS.len() - 1 {
            if self.custom_val_data.is_empty() {
                None
            } else {
                Some(self.custom_val_data.clone())
            }
        } else {
            Some(VAL_DATASETS[self.selected_val_idx].0.to_string())
        }
    }

    pub fn get_test_data(&self) -> String {
        if self.selected_test_idx == TEST_DATASETS.len() - 1 {
            self.custom_test_data.clone()
        } else {
            TEST_DATASETS[self.selected_test_idx].0.to_string()
        }
    }

    pub fn get_device(&self) -> String {
        HARDWARE_DEVICES[self.selected_hardware_idx].0.to_string()
    }

    pub fn get_output_dir(&self) -> String {
        if self.output_dir.is_empty() {
            let model = self.get_model();
            let model_name = model.replace('/', "-");
            format!("./output/{}-lora", model_name)
        } else {
            self.output_dir.clone()
        }
    }

    // ========================================================================
    // Input Handling
    // ========================================================================

    pub fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        // Handle editing modes first
        if self.editing_model {
            self.handle_text_input(key, &mut self.custom_model.clone(), |s, v| s.custom_model = v, |s| s.editing_model = false);
            return;
        }
        if self.editing_train_data {
            self.handle_text_input(key, &mut self.custom_train_data.clone(), |s, v| s.custom_train_data = v, |s| s.editing_train_data = false);
            return;
        }
        if self.editing_val_data {
            self.handle_text_input(key, &mut self.custom_val_data.clone(), |s, v| s.custom_val_data = v, |s| s.editing_val_data = false);
            return;
        }
        if self.editing_test_data {
            self.handle_text_input(key, &mut self.custom_test_data.clone(), |s, v| s.custom_test_data = v, |s| s.editing_test_data = false);
            return;
        }
        if self.editing_output {
            self.handle_text_input(key, &mut self.output_dir.clone(), |s, v| s.output_dir = v, |s| s.editing_output = false);
            return;
        }
        if self.editing_param {
            self.handle_param_input(key);
            return;
        }

        // Global keys
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
                    Section::Model => self.handle_model_key(key),
                    Section::TrainData => self.handle_train_data_key(key),
                    Section::ValData => self.handle_val_data_key(key),
                    Section::TestData => self.handle_test_data_key(key),
                    Section::Hardware => self.handle_hardware_key(key),
                    Section::Params => self.handle_params_key(key),
                    Section::Output => self.handle_output_key(key),
                    Section::Start => self.handle_start_key(key),
                }
            }
        }
    }

    fn handle_text_input<F, G>(&mut self, key: KeyCode, current: &mut String, set_value: F, finish: G)
    where
        F: FnOnce(&mut Self, String),
        G: FnOnce(&mut Self),
    {
        match key {
            KeyCode::Enter | KeyCode::Esc => {
                finish(self);
            }
            KeyCode::Backspace => {
                current.pop();
                let v = current.clone();
                set_value(self, v);
            }
            KeyCode::Char(c) => {
                current.push(c);
                let v = current.clone();
                set_value(self, v);
            }
            _ => {}
        }
    }

    fn handle_model_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_model_idx > 0 {
                    self.selected_model_idx -= 1;
                    self.model_list_state.select(Some(self.selected_model_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_model_idx < AVAILABLE_MODELS.len() - 1 {
                    self.selected_model_idx += 1;
                    self.model_list_state.select(Some(self.selected_model_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_model_idx == AVAILABLE_MODELS.len() - 1 {
                    self.editing_model = true;
                }
            }
            _ => {}
        }
    }

    fn handle_train_data_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_train_idx > 0 {
                    self.selected_train_idx -= 1;
                    self.train_data_list_state.select(Some(self.selected_train_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_train_idx < TRAIN_DATASETS.len() - 1 {
                    self.selected_train_idx += 1;
                    self.train_data_list_state.select(Some(self.selected_train_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_train_idx == TRAIN_DATASETS.len() - 1 {
                    self.editing_train_data = true;
                }
            }
            _ => {}
        }
    }

    fn handle_val_data_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_val_idx > 0 {
                    self.selected_val_idx -= 1;
                    self.val_data_list_state.select(Some(self.selected_val_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_val_idx < VAL_DATASETS.len() - 1 {
                    self.selected_val_idx += 1;
                    self.val_data_list_state.select(Some(self.selected_val_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_val_idx == VAL_DATASETS.len() - 1 {
                    self.editing_val_data = true;
                }
            }
            _ => {}
        }
    }

    fn handle_test_data_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Up => {
                if self.selected_test_idx > 0 {
                    self.selected_test_idx -= 1;
                    self.test_data_list_state.select(Some(self.selected_test_idx));
                }
            }
            KeyCode::Down => {
                if self.selected_test_idx < TEST_DATASETS.len() - 1 {
                    self.selected_test_idx += 1;
                    self.test_data_list_state.select(Some(self.selected_test_idx));
                }
            }
            KeyCode::Enter => {
                if self.selected_test_idx == TEST_DATASETS.len() - 1 {
                    self.editing_test_data = true;
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

    fn handle_params_key(&mut self, key: KeyCode) {
        let param_count = ParamField::all().len();
        match key {
            KeyCode::Up => {
                if self.selected_param_idx > 0 {
                    self.selected_param_idx -= 1;
                }
            }
            KeyCode::Down => {
                if self.selected_param_idx < param_count - 1 {
                    self.selected_param_idx += 1;
                }
            }
            KeyCode::Enter => {
                self.editing_param = true;
                self.param_input = self.get_param_value(self.selected_param_idx);
            }
            _ => {}
        }
    }

    fn handle_param_input(&mut self, key: KeyCode) {
        match key {
            KeyCode::Enter => {
                self.apply_param_value();
                self.editing_param = false;
                self.param_input.clear();
            }
            KeyCode::Esc => {
                self.editing_param = false;
                self.param_input.clear();
            }
            KeyCode::Backspace => {
                self.param_input.pop();
            }
            KeyCode::Char(c) => {
                self.param_input.push(c);
            }
            _ => {}
        }
    }

    fn get_param_value(&self, idx: usize) -> String {
        match ParamField::all()[idx] {
            ParamField::Epochs => self.epochs.to_string(),
            ParamField::LearningRate => format!("{:.0e}", self.learning_rate),
            ParamField::BatchSize => self.batch_size.to_string(),
            ParamField::LoraRank => self.lora_rank.to_string(),
            ParamField::LoraAlpha => self.lora_alpha.to_string(),
            ParamField::MaxSeqLength => self.max_seq_length.to_string(),
            ParamField::GradientAccum => self.gradient_accum.to_string(),
            ParamField::WarmupRatio => self.warmup_ratio.to_string(),
            ParamField::SaveSteps => self.save_steps.to_string(),
            ParamField::LoggingSteps => self.logging_steps.to_string(),
        }
    }

    fn apply_param_value(&mut self) {
        let field = ParamField::all()[self.selected_param_idx];
        let value = &self.param_input;

        match field {
            ParamField::Epochs => {
                if let Ok(v) = value.parse() { self.epochs = v; }
            }
            ParamField::LearningRate => {
                if let Ok(v) = value.parse() { self.learning_rate = v; }
            }
            ParamField::BatchSize => {
                if let Ok(v) = value.parse() { self.batch_size = v; }
            }
            ParamField::LoraRank => {
                if let Ok(v) = value.parse() { self.lora_rank = v; }
            }
            ParamField::LoraAlpha => {
                if let Ok(v) = value.parse() { self.lora_alpha = v; }
            }
            ParamField::MaxSeqLength => {
                if let Ok(v) = value.parse() { self.max_seq_length = v; }
            }
            ParamField::GradientAccum => {
                if let Ok(v) = value.parse() { self.gradient_accum = v; }
            }
            ParamField::WarmupRatio => {
                if let Ok(v) = value.parse() { self.warmup_ratio = v; }
            }
            ParamField::SaveSteps => {
                if let Ok(v) = value.parse() { self.save_steps = v; }
            }
            ParamField::LoggingSteps => {
                if let Ok(v) = value.parse() { self.logging_steps = v; }
            }
        }
    }

    fn handle_output_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            self.editing_output = true;
        }
    }

    fn handle_start_key(&mut self, key: KeyCode) {
        if key == KeyCode::Enter {
            if self.validate_config() {
                self.should_start = true;
            }
        }
    }

    fn validate_config(&mut self) -> bool {
        let model = self.get_model();
        if model.is_empty() {
            self.error_message = Some("Model name cannot be empty".to_string());
            return false;
        }

        let train_data = self.get_train_data();
        if train_data.is_empty() {
            self.error_message = Some("Training data path cannot be empty".to_string());
            return false;
        }

        let test_data = self.get_test_data();
        if test_data.is_empty() {
            self.error_message = Some("Test data path cannot be empty".to_string());
            return false;
        }

        if self.epochs == 0 {
            self.error_message = Some("Epochs must be > 0".to_string());
            return false;
        }

        if self.batch_size == 0 {
            self.error_message = Some("Batch size must be > 0".to_string());
            return false;
        }

        self.error_message = None;
        true
    }
}

// ============================================================================
// Drawing Functions
// ============================================================================

pub fn draw(f: &mut Frame, app: &mut TrainingApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    draw_header(f, chunks[0]);
    draw_tabs(f, app, chunks[1]);

    match app.state {
        AppState::Configuring => draw_config(f, app, chunks[2]),
        AppState::Training => draw_training(f, app, chunks[2]),
        AppState::Evaluating => draw_evaluating(f, app, chunks[2]),
        AppState::Results => draw_results(f, app, chunks[2]),
    }

    draw_footer(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, area: Rect) {
    let title = Line::from(vec![
        Span::styled("Rust", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled("RAG", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("Lab", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled("Training & Evaluation Pipeline", Style::default().fg(Color::Cyan)),
    ]);

    let header = Paragraph::new(title)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)))
        .alignment(Alignment::Center);

    f.render_widget(header, area);
}

fn draw_tabs(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let titles: Vec<Line> = Section::all()
        .iter()
        .map(|s| Line::from(s.name()))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Sections"))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .select(app.current_section.index());

    f.render_widget(tabs, area);
}

fn draw_config(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // Left panel: Model and Data selection
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(25), // Model
            Constraint::Percentage(25), // Train Data
            Constraint::Percentage(25), // Val Data
            Constraint::Percentage(25), // Test Data
        ])
        .split(main_chunks[0]);

    draw_model_section(f, app, left_chunks[0]);
    draw_train_data_section(f, app, left_chunks[1]);
    draw_val_data_section(f, app, left_chunks[2]);
    draw_test_data_section(f, app, left_chunks[3]);

    // Right panel: Hardware, Params, Output, Preview
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // Hardware
            Constraint::Min(10),    // Parameters
            Constraint::Length(5),  // Output
            Constraint::Length(8),  // Preview
        ])
        .split(main_chunks[1]);

    draw_hardware_section(f, app, right_chunks[0]);
    draw_params_section(f, app, right_chunks[1]);
    draw_output_section(f, app, right_chunks[2]);
    draw_preview_section(f, app, right_chunks[3]);
}

fn draw_model_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::Model;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = AVAILABLE_MODELS
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let content = if i == AVAILABLE_MODELS.len() - 1 && !app.custom_model.is_empty() {
                format!("{}: {}", name, app.custom_model)
            } else {
                format!("{} - {}", name, desc)
            };
            let style = if i == app.selected_model_idx {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(content)).style(style)
        })
        .collect();

    let title = if app.editing_model { "Model [EDITING]" } else { "Model" };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.model_list_state);
}

fn draw_train_data_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::TrainData;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = TRAIN_DATASETS
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let content = if i == TRAIN_DATASETS.len() - 1 && !app.custom_train_data.is_empty() {
                format!("{}: {}", name, app.custom_train_data)
            } else {
                format!("{} - {}", name, desc)
            };
            let style = if i == app.selected_train_idx {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(content)).style(style)
        })
        .collect();

    let title = if app.editing_train_data { "Training Data [EDITING]" } else { "Training Data" };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.train_data_list_state);
}

fn draw_val_data_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::ValData;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = VAL_DATASETS
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let content = if i == VAL_DATASETS.len() - 1 && !app.custom_val_data.is_empty() {
                format!("{}: {}", name, app.custom_val_data)
            } else {
                format!("{} - {}", name, desc)
            };
            let style = if i == app.selected_val_idx {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(content)).style(style)
        })
        .collect();

    let title = if app.editing_val_data { "Validation Data [EDITING]" } else { "Validation Data (optional)" };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.val_data_list_state);
}

fn draw_test_data_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::TestData;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = TEST_DATASETS
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let content = if i == TEST_DATASETS.len() - 1 && !app.custom_test_data.is_empty() {
                format!("{}: {}", name, app.custom_test_data)
            } else {
                format!("{} - {}", name, desc)
            };
            let style = if i == app.selected_test_idx {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(content)).style(style)
        })
        .collect();

    let title = if app.editing_test_data { "Test Data [EDITING]" } else { "Test Data (for evaluation)" };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.test_data_list_state);
}

fn draw_hardware_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::Hardware;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = HARDWARE_DEVICES
        .iter()
        .enumerate()
        .map(|(i, (id, name, desc))| {
            let status = get_device_status(id);
            let status_color = if status == "Available" { Color::Green } else { Color::Red };

            let style = if i == app.selected_hardware_idx {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let line = Line::from(vec![
                Span::styled(format!("{:<6}", name), style),
                Span::raw(" - "),
                Span::styled(format!("[{}]", status), Style::default().fg(status_color)),
                Span::raw(format!(" {}", desc)),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title("Hardware Acceleration"))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut app.hardware_list_state);
}

fn get_device_status(device: &str) -> &'static str {
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

fn draw_params_section(f: &mut Frame, app: &mut TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::Params;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let items: Vec<ListItem> = ParamField::all()
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let value = if app.editing_param && i == app.selected_param_idx {
                format!("{}_", app.param_input)
            } else {
                app.get_param_value(i)
            };

            let style = if i == app.selected_param_idx && is_active {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let content = format!("{:<14}: {}", field.name(), value);
            ListItem::new(Line::from(content)).style(style)
        })
        .collect();

    let title = if app.editing_param { "Parameters [EDITING]" } else { "Parameters (Enter to edit)" };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title));

    f.render_widget(list, area);
}

fn draw_output_section(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::Output;
    let border_color = if is_active { Color::Yellow } else { Color::White };

    let output_display = if app.editing_output {
        format!("{}_", app.output_dir)
    } else {
        app.get_output_dir()
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("Path: ", Style::default().fg(Color::Cyan)),
            Span::raw(&output_display),
        ]),
        Line::from(vec![
            Span::styled("(Press Enter to edit)", Style::default().fg(Color::Gray)),
        ]),
    ];

    let title = if app.editing_output { "Output Directory [EDITING]" } else { "Output Directory" };
    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title(title));

    f.render_widget(para, area);
}

fn draw_preview_section(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let is_active = app.current_section == Section::Start;
    let border_color = if is_active { Color::Green } else { Color::White };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_model()),
        ]),
        Line::from(vec![
            Span::styled("Train: ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_train_data()),
        ]),
        Line::from(vec![
            Span::styled("Device: ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_device()),
            Span::raw(format!("  |  Epochs: {}  |  Batch: {}", app.epochs, app.batch_size)),
        ]),
    ];

    if is_active {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(">>> Press ENTER to start training <<<", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]));
    }

    if let Some(ref err) = app.error_message {
        lines.push(Line::from(vec![
            Span::styled(format!("Error: {}", err), Style::default().fg(Color::Red)),
        ]));
    }

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color)).title("Configuration Preview"))
        .wrap(Wrap { trim: true });

    f.render_widget(para, area);
}

fn draw_training(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let progress = &app.progress;

    let epoch_pct = if progress.total_epochs > 0 {
        (progress.epoch as f64 / progress.total_epochs as f64 * 100.0) as u16
    } else { 0 };

    let step_pct = if progress.total_steps > 0 {
        (progress.step as f64 / progress.total_steps as f64 * 100.0) as u16
    } else { 0 };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Info
            Constraint::Length(3),  // Epoch gauge
            Constraint::Length(3),  // Step gauge
            Constraint::Min(5),     // Metrics
        ])
        .margin(1)
        .split(area);

    // Info
    let info_lines = vec![
        Line::from(vec![
            Span::styled("Model:   ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_model()),
        ]),
        Line::from(vec![
            Span::styled("Dataset: ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_train_data()),
        ]),
        Line::from(vec![
            Span::styled("Device:  ", Style::default().fg(Color::Cyan)),
            Span::raw(app.get_device()),
        ]),
    ];
    let info = Paragraph::new(info_lines)
        .block(Block::default().borders(Borders::ALL).title("Training in Progress"));
    f.render_widget(info, chunks[0]);

    // Epoch gauge
    let epoch_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!("Epoch {}/{}", progress.epoch, progress.total_epochs)))
        .gauge_style(Style::default().fg(Color::Yellow))
        .percent(epoch_pct);
    f.render_widget(epoch_gauge, chunks[1]);

    // Step gauge
    let step_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!("Step {}/{}", progress.step, progress.total_steps)))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(step_pct);
    f.render_widget(step_gauge, chunks[2]);

    // Metrics
    let mut metric_lines = vec![
        Line::from(vec![
            Span::styled("Training Loss:  ", Style::default().fg(Color::Green)),
            Span::raw(format!("{:.4}", progress.train_loss)),
        ]),
        Line::from(vec![
            Span::styled("Learning Rate:  ", Style::default().fg(Color::Green)),
            Span::raw(format!("{:.2e}", progress.learning_rate)),
        ]),
        Line::from(vec![
            Span::styled("Throughput:     ", Style::default().fg(Color::Green)),
            Span::raw(format!("{:.1} samples/sec", progress.samples_per_sec)),
        ]),
    ];

    if let Some(val_loss) = progress.val_loss {
        metric_lines.push(Line::from(vec![
            Span::styled("Validation Loss: ", Style::default().fg(Color::Blue)),
            Span::raw(format!("{:.4}", val_loss)),
        ]));
    }

    if !progress.status_message.is_empty() {
        metric_lines.push(Line::from(""));
        metric_lines.push(Line::from(vec![
            Span::styled(&progress.status_message, Style::default().fg(Color::Gray)),
        ]));
    }

    let metrics = Paragraph::new(metric_lines)
        .block(Block::default().borders(Borders::ALL).title("Metrics"));
    f.render_widget(metrics, chunks[3]);
}

fn draw_evaluating(f: &mut Frame, _app: &TrainingApp, area: Rect) {
    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Running evaluation on test dataset...", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Please wait...", Style::default().fg(Color::Gray)),
        ]),
    ];

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Evaluating"))
        .alignment(Alignment::Center);

    f.render_widget(para, area);
}

fn draw_results(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let results = &app.results;

    let status_icon = if results.success { "✓" } else { "✗" };
    let status_color = if results.success { Color::Green } else { Color::Red };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{} Training {}", status_icon, if results.success { "completed successfully" } else { "failed" }),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        ]),
    ];

    if results.success {
        lines.push(Line::from(vec![
            Span::styled("Checkpoint: ", Style::default().fg(Color::Cyan)),
            Span::raw(&results.checkpoint_path),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("═══════════ Evaluation Results ═══════════", Style::default().fg(Color::Yellow)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(format!("  ACCURACY: {:.1}%  ({}/{} correct)",
                results.test_accuracy * 100.0, results.test_correct, results.test_total),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]));

        if let Some(base_acc) = results.base_accuracy {
            let improvement = results.test_accuracy - base_acc;
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("  Base Model:      ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1}%", base_acc * 100.0)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Fine-tuned:      ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.1}%  ({:+.1}%)", results.test_accuracy * 100.0, improvement * 100.0)),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Training Time:    ", Style::default().fg(Color::Gray)),
            Span::raw(format_duration(results.training_time_secs)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Final Train Loss: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{:.4}", results.final_train_loss)),
        ]));

        if let Some(val_loss) = results.final_val_loss {
            lines.push(Line::from(vec![
                Span::styled("Final Val Loss:   ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{:.4}", val_loss)),
            ]));
        }
    } else if let Some(ref err) = results.error_message {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(format!("Error: {}", err), Style::default().fg(Color::Red)),
        ]));
    }

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(status_color)).title("Results"))
        .alignment(Alignment::Center);

    f.render_widget(para, area);
}

fn format_duration(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, mins, secs)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs)
    } else {
        format!("{}s", secs)
    }
}

fn draw_footer(f: &mut Frame, app: &TrainingApp, area: Rect) {
    let help_text = match app.state {
        AppState::Configuring => {
            if app.editing_model || app.editing_train_data || app.editing_val_data ||
               app.editing_test_data || app.editing_param || app.editing_output {
                "[Enter] Confirm  [Esc] Cancel"
            } else {
                "[Tab] Next section  [↑↓] Navigate  [Enter] Select/Edit  [q] Quit"
            }
        }
        AppState::Training => "[Ctrl+C] Cancel training",
        AppState::Evaluating => "Evaluation in progress...",
        AppState::Results => "[Enter] New training  [q] Quit",
    };

    let footer = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

    f.render_widget(footer, area);
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Configuration returned from the TUI
#[derive(Debug, Clone)]
pub struct TrainingAppConfig {
    pub model: String,
    pub train_data: String,
    pub val_data: Option<String>,
    pub test_data: String,
    pub output_dir: String,
    pub device: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub max_seq_length: usize,
    pub gradient_accumulation: usize,
    pub warmup_ratio: f64,
    pub save_steps: usize,
    pub logging_steps: usize,
}

/// Run the training configuration TUI
pub fn run_training_tui() -> Result<Option<TrainingAppConfig>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = TrainingApp::new();
    let tick_rate = Duration::from_millis(100);

    loop {
        terminal.draw(|f| draw(f, &mut app))?;

        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                app.handle_key(key.code, key.modifiers);
            }
        }

        if app.should_quit {
            break;
        }

        if app.should_start {
            let config = TrainingAppConfig {
                model: app.get_model(),
                train_data: app.get_train_data(),
                val_data: app.get_val_data(),
                test_data: app.get_test_data(),
                output_dir: app.get_output_dir(),
                device: app.get_device(),
                epochs: app.epochs,
                learning_rate: app.learning_rate,
                batch_size: app.batch_size,
                lora_rank: app.lora_rank,
                lora_alpha: app.lora_alpha,
                max_seq_length: app.max_seq_length,
                gradient_accumulation: app.gradient_accum,
                warmup_ratio: app.warmup_ratio,
                save_steps: app.save_steps,
                logging_steps: app.logging_steps,
            };

            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;

            return Ok(Some(config));
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(None)
}
