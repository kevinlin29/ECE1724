//! TUI Application State

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Active view/tab in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Pipeline,
    Query,
    Stats,
    Logs,
}

impl View {
    pub fn title(&self) -> &str {
        match self {
            View::Pipeline => "Pipeline",
            View::Query => "Query",
            View::Stats => "Statistics",
            View::Logs => "Logs",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            View::Pipeline => 0,
            View::Query => 1,
            View::Stats => 2,
            View::Logs => 3,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => View::Pipeline,
            1 => View::Query,
            2 => View::Stats,
            _ => View::Logs,
        }
    }
}

/// Pipeline stage status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageStatus {
    NotStarted,
    Running,
    Completed,
    Failed(String),
}

/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub name: String,
    pub status: StageStatus,
    pub progress: f64, // 0.0 to 1.0
    pub details: String,
}

/// Query result for display
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rank: usize,
    pub chunk_id: String,
    pub document_id: String,
    pub score: f32,
    pub content: String,
}

/// Application state
pub struct App {
    /// Current active view
    pub current_view: View,

    /// Should the app quit?
    pub should_quit: bool,

    /// Pipeline stages
    pub stages: Vec<PipelineStage>,

    /// Query input text
    pub query_input: String,

    /// Query results
    pub query_results: Vec<QueryResult>,

    /// Whether query is running
    pub query_running: bool,

    /// Project paths
    pub chunks_dir: Option<PathBuf>,
    pub embeddings_dir: Option<PathBuf>,
    pub index_dir: Option<PathBuf>,

    /// Statistics
    pub num_documents: usize,
    pub num_chunks: usize,
    pub num_embeddings: usize,
    pub embedding_dimension: usize,

    /// Logs
    pub logs: Vec<String>,

    /// Last update time
    pub last_update: Instant,

    /// Cursor position for query input
    pub cursor_position: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            current_view: View::Pipeline,
            should_quit: false,
            stages: vec![
                PipelineStage {
                    name: "Ingest".to_string(),
                    status: StageStatus::NotStarted,
                    progress: 0.0,
                    details: "Load and chunk documents".to_string(),
                },
                PipelineStage {
                    name: "Embed".to_string(),
                    status: StageStatus::NotStarted,
                    progress: 0.0,
                    details: "Generate embeddings".to_string(),
                },
                PipelineStage {
                    name: "Index".to_string(),
                    status: StageStatus::NotStarted,
                    progress: 0.0,
                    details: "Build retrieval indexes".to_string(),
                },
                PipelineStage {
                    name: "Query".to_string(),
                    status: StageStatus::NotStarted,
                    progress: 0.0,
                    details: "Search and retrieve".to_string(),
                },
            ],
            query_input: String::new(),
            query_results: Vec::new(),
            query_running: false,
            chunks_dir: None,
            embeddings_dir: None,
            index_dir: None,
            num_documents: 0,
            num_chunks: 0,
            num_embeddings: 0,
            embedding_dimension: 384,
            logs: Vec::new(),
            last_update: Instant::now(),
            cursor_position: 0,
        }
    }
}

impl App {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle keyboard input
    pub fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        match self.current_view {
            View::Query => self.handle_query_key(key, modifiers),
            _ => self.handle_global_key(key, modifiers),
        }
    }

    fn handle_global_key(&mut self, key: KeyCode, _modifiers: KeyModifiers) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('1') => self.current_view = View::Pipeline,
            KeyCode::Char('2') => self.current_view = View::Query,
            KeyCode::Char('3') => self.current_view = View::Stats,
            KeyCode::Char('4') => self.current_view = View::Logs,
            KeyCode::Tab => {
                let next_index = (self.current_view.index() + 1) % 4;
                self.current_view = View::from_index(next_index);
            }
            _ => {}
        }
    }

    fn handle_query_key(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        match key {
            KeyCode::Char('q') if !modifiers.contains(KeyModifiers::NONE) => {
                self.should_quit = true;
            }
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
                // TODO: Execute query
                self.add_log(format!("Query: {}", self.query_input));
            }
            KeyCode::Tab => {
                let next_index = (self.current_view.index() + 1) % 4;
                self.current_view = View::from_index(next_index);
            }
            _ => {}
        }
    }

    /// Add a log message
    pub fn add_log(&mut self, message: String) {
        let timestamp = chrono::Local::now().format("%H:%M:%S");
        self.logs.push(format!("[{}] {}", timestamp, message));

        // Keep only last 100 logs
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }

    /// Update statistics from filesystem
    pub fn update_stats(&mut self) {
        // TODO: Scan directories and update counts
        self.last_update = Instant::now();
    }
}

/// Run the TUI application
pub fn run_tui() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new();
    app.add_log("RustRAGLab TUI started".to_string());
    app.add_log("Press 'q' to quit, 1-4 to switch views, Tab to cycle".to_string());

    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    // Main loop
    loop {
        terminal.draw(|f| super::ui::draw(f, &mut app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                app.handle_key(key.code, key.modifiers);
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // Update tick
            last_tick = Instant::now();

            // Periodic updates
            if app.last_update.elapsed() >= Duration::from_secs(5) {
                app.update_stats();
            }
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
