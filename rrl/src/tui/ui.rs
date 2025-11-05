//! TUI Rendering

use super::app::{App, StageStatus, View};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap,
    },
    Frame,
};

/// Main draw function
pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
        ])
        .split(f.area());

    draw_header(f, chunks[0]);
    draw_tabs(f, app, chunks[1]);

    match app.current_view {
        View::Pipeline => draw_pipeline(f, app, chunks[2]),
        View::Query => draw_query(f, app, chunks[2]),
        View::Stats => draw_stats(f, app, chunks[2]),
        View::Logs => draw_logs(f, app, chunks[2]),
    }
}

fn draw_header(f: &mut Frame, area: Rect) {
    let title = vec![
        Line::from(vec![
            Span::styled(
                "Rust",
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "RAG",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Lab",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                "Retrieval-Augmented Generation Framework",
                Style::default().fg(Color::Gray),
            ),
        ]),
    ];

    let header = Paragraph::new(title)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .alignment(Alignment::Center);

    f.render_widget(header, area);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = vec!["Pipeline", "Query", "Stats", "Logs"]
        .iter()
        .map(|t| Line::from(*t))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Views"))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .select(app.current_view.index());

    f.render_widget(tabs, area);
}

fn draw_pipeline(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    // Title
    let title = Paragraph::new("RAG Pipeline: Ingest → Embed → Index → Query")
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    // Draw each stage
    for (i, stage) in app.stages.iter().enumerate() {
        if i + 1 < chunks.len() {
            draw_stage(f, stage, chunks[i + 1]);
        }
    }
}

fn draw_stage(f: &mut Frame, stage: &super::app::PipelineStage, area: Rect) {
    let color = match stage.status {
        StageStatus::NotStarted => Color::Gray,
        StageStatus::Running => Color::Yellow,
        StageStatus::Completed => Color::Green,
        StageStatus::Failed(_) => Color::Red,
    };

    let status_icon = match stage.status {
        StageStatus::NotStarted => "⭘",
        StageStatus::Running => "⟳",
        StageStatus::Completed => "✓",
        StageStatus::Failed(_) => "✗",
    };

    let title = format!("{} {} - {}", status_icon, stage.name, stage.details);

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(color))
        .percent((stage.progress * 100.0) as u16);

    f.render_widget(gauge, area);
}

fn draw_query(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Input
            Constraint::Min(0),    // Results
        ])
        .split(area);

    // Query input
    let input = Paragraph::new(app.query_input.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Query (press Enter to search)"),
        )
        .style(Style::default().fg(Color::Yellow));

    f.render_widget(input, chunks[0]);

    // Cursor
    f.set_cursor_position((
        chunks[0].x + app.cursor_position as u16 + 1,
        chunks[0].y + 1,
    ));

    // Results
    if app.query_results.is_empty() {
        let placeholder = Paragraph::new(
            "No results yet.\n\nEnter a query above and press Enter to search.\n\nExample: How does Rust ensure memory safety?"
        )
        .block(Block::default().borders(Borders::ALL).title("Results"))
        .style(Style::default().fg(Color::Gray))
        .wrap(Wrap { trim: true });

        f.render_widget(placeholder, chunks[1]);
    } else {
        draw_query_results(f, app, chunks[1]);
    }
}

fn draw_query_results(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .query_results
        .iter()
        .map(|result| {
            let content = vec![
                Line::from(vec![
                    Span::styled(
                        format!("#{} ", result.rank),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("(score: {:.4}) ", result.score),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(&result.chunk_id),
                ]),
                Line::from(vec![
                    Span::styled("  Doc: ", Style::default().fg(Color::Gray)),
                    Span::raw(&result.document_id),
                ]),
                Line::from(vec![
                    Span::styled("  ", Style::default()),
                    Span::raw(
                        result
                            .content
                            .chars()
                            .take(100)
                            .collect::<String>(),
                    ),
                    Span::raw(if result.content.len() > 100 {
                        "..."
                    } else {
                        ""
                    }),
                ]),
                Line::from(""),
            ];
            ListItem::new(content)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!("Results ({})", app.query_results.len())),
    );

    f.render_widget(list, area);
}

fn draw_stats(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);

    // Data statistics
    let data_stats = vec![
        Line::from(vec![
            Span::styled("Documents:       ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", app.num_documents)),
        ]),
        Line::from(vec![
            Span::styled("Chunks:          ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", app.num_chunks)),
        ]),
        Line::from(vec![
            Span::styled("Embeddings:      ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", app.num_embeddings)),
        ]),
        Line::from(vec![
            Span::styled("Dimension:       ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", app.embedding_dimension)),
        ]),
    ];

    let data_widget = Paragraph::new(data_stats)
        .block(Block::default().borders(Borders::ALL).title("Data"))
        .style(Style::default().fg(Color::White));

    f.render_widget(data_widget, chunks[0]);

    // Paths
    let paths = vec![
        Line::from(vec![
            Span::styled("Chunks:     ", Style::default().fg(Color::Cyan)),
            Span::raw(
                app.chunks_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "Not set".to_string()),
            ),
        ]),
        Line::from(vec![
            Span::styled("Embeddings: ", Style::default().fg(Color::Cyan)),
            Span::raw(
                app.embeddings_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "Not set".to_string()),
            ),
        ]),
        Line::from(vec![
            Span::styled("Index:      ", Style::default().fg(Color::Cyan)),
            Span::raw(
                app.index_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "Not set".to_string()),
            ),
        ]),
    ];

    let paths_widget = Paragraph::new(paths)
        .block(Block::default().borders(Borders::ALL).title("Paths"))
        .style(Style::default().fg(Color::White));

    f.render_widget(paths_widget, chunks[1]);

    // Help
    let help = vec![
        Line::from("Keyboard Shortcuts:"),
        Line::from(""),
        Line::from(vec![
            Span::styled("1-4  ", Style::default().fg(Color::Yellow)),
            Span::raw("Switch views"),
        ]),
        Line::from(vec![
            Span::styled("Tab  ", Style::default().fg(Color::Yellow)),
            Span::raw("Cycle views"),
        ]),
        Line::from(vec![
            Span::styled("q    ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
    ];

    let help_widget = Paragraph::new(help)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .style(Style::default().fg(Color::Gray));

    f.render_widget(help_widget, chunks[2]);
}

fn draw_logs(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .rev()
        .map(|log| ListItem::new(Line::from(log.as_str())))
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Logs (most recent first)"),
    );

    f.render_widget(list, area);
}
