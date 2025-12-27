//! TUI Rendering Module
//!
//! Renders the wizard UI using ratatui widgets.

use crate::tui::app::{App, WizardStep};
#[allow(unused_imports)]
use crate::tui::detection::ProviderStatus;
use crate::tui::health::CheckStatus;
use crate::tui::indexer::{DataSetupOption, DataSetupSubStep, ImportMode};
use ratatui::prelude::*;
use ratatui::widgets::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn render(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Footer/help
        ])
        .split(frame.area());

    render_header(frame, chunks[0], app);
    render_main(frame, chunks[1], app);
    render_footer(frame, chunks[2], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let title = format!(
        " rmcp_memex wizard v{} - Step {}/{}: {} ",
        VERSION,
        app.step.step_number(),
        WizardStep::total_steps(),
        app.step.title()
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan).bold());

    frame.render_widget(block, area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let help_text = if app.input_mode || app.data_setup.input_mode {
        " [Enter] Confirm | [Esc] Cancel | Type to edit "
    } else {
        match app.step {
            WizardStep::Welcome => " [->/n] Next | [q] Quit ",
            WizardStep::EmbedderSetup => {
                " [Up/Down] Navigate | [Enter] Select | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::MemexSettings => {
                " [Up/Down] Navigate | [Enter] Edit | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::HostSelection => {
                " [Up/Down] Navigate | [Space] Toggle | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::SnippetPreview => " [->] Next | [<-] Back | [q] Quit ",
            WizardStep::HealthCheck => {
                if app.health_running {
                    " Running health checks... "
                } else {
                    " [Enter/r] Run Check | [->] Next | [<-] Back | [q] Quit "
                }
            }
            WizardStep::DataSetup => match app.data_setup.sub_step {
                DataSetupSubStep::SelectOption => {
                    " [Up/Down] Select | [Enter] Choose | [->] Skip | [<-] Back | [q] Quit "
                }
                DataSetupSubStep::EnterPath | DataSetupSubStep::EnterNamespace => {
                    " [Enter] Confirm | [Esc] Cancel "
                }
                DataSetupSubStep::SelectImportMode => {
                    " [Up/Down] Select | [Enter] Choose | [<-] Back "
                }
                DataSetupSubStep::Indexing => " Indexing in progress... ",
                DataSetupSubStep::Complete => " [->] Next | [<-] Back | [q] Quit ",
            },
            WizardStep::Summary => " [Enter] Write Configs | [<-] Back | [q] Quit ",
        }
    };

    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        );

    frame.render_widget(help, area);
}

fn render_main(frame: &mut Frame, area: Rect, app: &App) {
    match app.step {
        WizardStep::Welcome => render_welcome(frame, area, app),
        WizardStep::EmbedderSetup => render_embedder_setup(frame, area, app),
        WizardStep::MemexSettings => render_settings(frame, area, app),
        WizardStep::HostSelection => render_host_selection(frame, area, app),
        WizardStep::SnippetPreview => render_snippet_preview(frame, area, app),
        WizardStep::HealthCheck => render_health_check(frame, area, app),
        WizardStep::DataSetup => render_data_setup(frame, area, app),
        WizardStep::Summary => render_summary(frame, area, app),
    }
}

fn render_welcome(frame: &mut Frame, area: Rect, app: &App) {
    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Welcome to rmcp_memex Configuration Wizard",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from("This wizard will help you:"),
        Line::from(""),
        Line::from("  1. Configure rmcp_memex settings (database path, cache, etc.)"),
        Line::from("  2. Detect and configure MCP host integrations"),
        Line::from("  3. Generate configuration snippets for your hosts"),
        Line::from("  4. Verify your setup with a health check"),
        Line::from(""),
        Line::from(Span::styled(
            format!("Binary: {}", app.binary_path),
            Style::default().fg(Color::Green),
        )),
        Line::from(""),
        Line::from(format!(
            "Detected {} MCP host configuration(s)",
            app.hosts.iter().filter(|h| h.exists).count()
        )),
        Line::from(""),
        if app.dry_run {
            Line::from(Span::styled(
                "DRY RUN MODE - No files will be written",
                Style::default().fg(Color::Yellow).bold(),
            ))
        } else {
            Line::from("")
        },
        Line::from(""),
        Line::from(Span::styled(
            "Press → or 'n' to continue...",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text).alignment(Alignment::Center).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );

    frame.render_widget(paragraph, area);
}

fn render_settings(frame: &mut Frame, area: Rect, app: &App) {
    let fields = [
        (
            "Database Path",
            &app.memex_cfg.db_path,
            "LanceDB storage location",
        ),
        (
            "Cache Size (MB)",
            &app.memex_cfg.cache_mb.to_string(),
            "In-memory cache size",
        ),
        (
            "Log Level",
            &app.memex_cfg.log_level,
            "trace/debug/info/warn/error",
        ),
        (
            "Max Request (bytes)",
            &app.memex_cfg.max_request_bytes.to_string(),
            "JSON-RPC size limit",
        ),
        ("Mode", &app.memex_cfg.mode, "full or memory"),
    ];

    let items: Vec<ListItem> = fields
        .iter()
        .enumerate()
        .map(|(i, (label, value, hint))| {
            let is_focused = i == app.focus;
            let is_editing = app.input_mode && app.editing_field == Some(i);

            let display_value = if is_editing {
                format!("{}▏", app.input_buffer)
            } else {
                (*value).clone()
            };

            let style = if is_editing {
                Style::default().fg(Color::Yellow).bg(Color::DarkGray)
            } else if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else {
                Style::default()
            };

            let prefix = if is_focused { "▶ " } else { "  " };

            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(format!("{:<20}", label), style),
                Span::styled(format!("{:<40}", display_value), style),
                Span::styled(format!(" ({})", hint), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Memex Configuration "),
    );

    frame.render_widget(list, area);
}

fn render_host_selection(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .hosts
        .iter()
        .enumerate()
        .map(|(i, host)| {
            let is_focused = i == app.focus;
            let is_selected = app.selected_hosts.contains(&i);

            let checkbox = if is_selected { "[x]" } else { "[ ]" };
            let status = host.status_icon();

            let style = if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else if is_selected {
                Style::default().fg(Color::Green)
            } else if !host.exists {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default()
            };

            let prefix = if is_focused { "▶ " } else { "  " };

            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(format!("{} ", checkbox), style),
                Span::styled(format!("{} ", status), style),
                Span::styled(format!("{:<20}", host.kind.display_name()), style),
                Span::styled(
                    format!(" {} ", host.status_text()),
                    if host.has_rmcp_memex {
                        Style::default().fg(Color::Green)
                    } else if host.exists {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Select MCP Hosts to Configure "),
    );

    frame.render_widget(list, area);
}

fn render_snippet_preview(frame: &mut Frame, area: Rect, app: &App) {
    let snippets = app.generate_snippets();

    let mut lines = vec![
        Line::from(Span::styled(
            "Configuration snippets for selected hosts:",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
    ];

    if snippets.is_empty() {
        lines.push(Line::from(Span::styled(
            "No hosts selected. Go back and select at least one host.",
            Style::default().fg(Color::Yellow),
        )));
    } else {
        for (kind, snippet) in snippets {
            lines.push(Line::from(Span::styled(
                format!("--- {} ---", kind.display_name()),
                Style::default().fg(Color::Green).bold(),
            )));
            for line in snippet.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", line),
                    Style::default().fg(Color::White),
                )));
            }
            lines.push(Line::from(""));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Config Snippets Preview "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_health_check(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Health Check",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
    ];

    // Show status line
    if app.health_running {
        lines.push(Line::from(Span::styled(
            "[...] Running health checks...",
            Style::default().fg(Color::Yellow),
        )));
    } else if let Some(status) = &app.health_status {
        lines.push(Line::from(Span::styled(
            status.clone(),
            if status.contains("passed") {
                Style::default().fg(Color::Green)
            } else if status.contains("failed") {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Yellow)
            },
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press [Enter] or [r] to run health checks",
            Style::default().fg(Color::Yellow),
        )));
    }

    lines.push(Line::from(""));

    // Show detailed health check results
    if let Some(ref result) = app.health_result {
        lines.push(Line::from(Span::styled("Checks:", Style::default().bold())));

        for item in &result.items {
            let (icon, color) = match &item.status {
                CheckStatus::Pass => ("[OK]", Color::Green),
                CheckStatus::Fail(_) => ("[ERR]", Color::Red),
                CheckStatus::Running => ("[...]", Color::Yellow),
                CheckStatus::Pending => ("[ ]", Color::DarkGray),
            };

            lines.push(Line::from(vec![
                Span::styled(format!("  {} ", icon), Style::default().fg(color)),
                Span::styled(&item.name, Style::default().fg(color)),
                Span::styled(
                    format!(" - {}", item.description),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));

            // Show error details
            if let CheckStatus::Fail(ref msg) = item.status {
                for error_line in msg.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("       {}", error_line),
                        Style::default().fg(Color::Red),
                    )));
                }
            }
        }

        lines.push(Line::from(""));

        // Show connected provider info
        if let Some(ref provider) = result.connected_provider {
            lines.push(Line::from(vec![
                Span::styled("  Connected to: ", Style::default()),
                Span::styled(provider, Style::default().fg(Color::Green).bold()),
            ]));
        }

        if let Some(dim) = result.verified_dimension {
            lines.push(Line::from(vec![
                Span::styled("  Vector dimension: ", Style::default()),
                Span::styled(dim.to_string(), Style::default().fg(Color::Cyan)),
            ]));
        }
    }

    // Show any additional messages
    if !app.messages.is_empty() {
        lines.push(Line::from(""));
        for msg in &app.messages {
            let style = if msg.starts_with("[OK]") {
                Style::default().fg(Color::Green)
            } else if msg.starts_with("[ERR]") {
                Style::default().fg(Color::Red)
            } else if msg.starts_with("[-]") {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(msg.clone(), style)));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Health Check "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_embedder_setup(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Embedding Provider Setup",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from("Configure which embedding provider to use for vector search."),
        Line::from(""),
    ];

    // Show current configuration
    lines.push(Line::from(Span::styled(
        "Current Configuration:",
        Style::default().bold(),
    )));
    lines.push(Line::from(format!(
        "  Provider: {}",
        app.embedding_config.provider_name()
    )));
    lines.push(Line::from(format!(
        "  Model: {}",
        app.embedding_config.model_name()
    )));
    lines.push(Line::from(format!(
        "  Dimension: {}",
        app.embedding_config.dimension()
    )));
    lines.push(Line::from(""));

    // Show detected providers if available
    if !app.embedder_state.detected_providers.is_empty() {
        lines.push(Line::from(Span::styled(
            "Detected Providers:",
            Style::default().bold(),
        )));

        for (i, provider) in app.embedder_state.detected_providers.iter().enumerate() {
            let is_focused = i == app.focus;
            let prefix = if is_focused { "▶ " } else { "  " };
            let style = if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else {
                Style::default()
            };

            let status_icon = match &provider.status {
                ProviderStatus::Online(_) => "[OK]",
                ProviderStatus::OnlineNoModel => "[--]",
                ProviderStatus::Offline => "[XX]",
            };

            let status_color = match &provider.status {
                ProviderStatus::Online(_) => Color::Green,
                ProviderStatus::OnlineNoModel => Color::Yellow,
                ProviderStatus::Offline => Color::DarkGray,
            };

            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(provider.display_name(), style),
            ]));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "No providers detected yet. Continue to proceed with default configuration.",
            Style::default().fg(Color::Yellow),
        )));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Embedder Setup "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_data_setup(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Data Setup",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
    ];

    match app.data_setup.sub_step {
        DataSetupSubStep::SelectOption => {
            lines.push(Line::from("Select how to initialize your data:"));
            lines.push(Line::from(""));

            for (i, option) in DataSetupOption::all().iter().enumerate() {
                let is_focused = i == app.data_setup.focus;
                let prefix = if is_focused { "▶ " } else { "  " };
                let style = if is_focused {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default()
                };

                lines.push(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(option.display_name(), style),
                ]));
                lines.push(Line::from(Span::styled(
                    format!("    {}", option.description()),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
        DataSetupSubStep::EnterPath => {
            lines.push(Line::from(format!(
                "Enter path for {:?}:",
                app.data_setup.option
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("> {}|", app.data_setup.input_buffer),
                Style::default().fg(Color::Yellow),
            )));
        }
        DataSetupSubStep::EnterNamespace => {
            lines.push(Line::from("Enter namespace for indexed documents:"));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("> {}|", app.data_setup.input_buffer),
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "(Leave empty for default 'rag' namespace)",
                Style::default().fg(Color::DarkGray),
            )));
        }
        DataSetupSubStep::SelectImportMode => {
            lines.push(Line::from("Select import mode:"));
            lines.push(Line::from(""));

            for (i, mode) in ImportMode::all().iter().enumerate() {
                let is_focused = i == app.data_setup.focus;
                let prefix = if is_focused { "▶ " } else { "  " };
                let style = if is_focused {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default()
                };

                lines.push(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(mode.display_name(), style),
                ]));
            }
        }
        DataSetupSubStep::Indexing => {
            lines.push(Line::from(Span::styled(
                "Indexing in progress...",
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(""));

            if let Some(ref progress) = app.data_setup.progress {
                let percentage = if progress.total > 0 {
                    (progress.processed as f64 / progress.total as f64 * 100.0) as u16
                } else {
                    0
                };

                lines.push(Line::from(format!(
                    "Progress: {}/{} files ({:.1}%)",
                    progress.processed, progress.total, percentage as f64
                )));

                if !progress.current_file.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!("Current: {}", progress.current_file),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
        }
        DataSetupSubStep::Complete => {
            lines.push(Line::from(Span::styled(
                "Data setup complete!",
                Style::default().fg(Color::Green).bold(),
            )));
            lines.push(Line::from(""));

            for msg in &app.messages {
                let style = if msg.starts_with("[OK]") {
                    Style::default().fg(Color::Green)
                } else if msg.starts_with("[ERR]") {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(msg.clone(), style)));
            }
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Data Setup "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_summary(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Configuration Summary",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled("Memex Settings:", Style::default().bold())),
        Line::from(format!("  Database: {}", app.memex_cfg.db_path)),
        Line::from(format!("  Cache: {} MB", app.memex_cfg.cache_mb)),
        Line::from(format!("  Log Level: {}", app.memex_cfg.log_level)),
        Line::from(format!("  Mode: {}", app.memex_cfg.mode)),
        Line::from(""),
        Line::from(Span::styled("Selected Hosts:", Style::default().bold())),
    ];

    let selected = app.get_selected_hosts();
    if selected.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none selected)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for host in selected {
            lines.push(Line::from(format!("  • {}", host.kind.display_name())));
        }
    }

    lines.push(Line::from(""));

    if app.dry_run {
        lines.push(Line::from(Span::styled(
            "DRY RUN MODE - Press [Enter] to preview changes",
            Style::default().fg(Color::Yellow).bold(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press [Enter] to write configuration files",
            Style::default().fg(Color::Green).bold(),
        )));
    }

    lines.push(Line::from(""));

    // Show any messages from write operation
    for msg in &app.messages {
        let style = if msg.starts_with('✓') {
            Style::default().fg(Color::Green)
        } else if msg.contains("Warning") || msg.contains("DRY RUN") {
            Style::default().fg(Color::Yellow)
        } else if msg.contains("Error") {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(Span::styled(msg.clone(), style)));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Summary "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}
