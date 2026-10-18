#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sampler;

use eframe::egui;
use egui::emath::GuiRounding;
use egui::scroll_area::ScrollBarVisibility;
use egui_extras::{Column, TableBuilder};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::mpsc,
    time::Duration,
};

const REFRESH: Duration = Duration::from_secs(1);
/// How many recent samples the background bars scale against and the peak sort looks back over.
const HISTORY: usize = 10;
/// The key eframe itself stores the window place under, which we reuse.
const WINDOW_KEY: &str = "window";
/// Where the last sorted column and its order are kept across runs.
const SORT_KEY: &str = "sort";
/// The background bar behind each value and total.
const BAR: egui::Color32 = egui::Color32::from_rgb(45, 75, 120);
/// Half-transparent `BAR`, for the recent-peak bar behind the current one.
const FAINT: egui::Color32 = egui::Color32::from_rgba_premultiplied(23, 38, 60, 128);
/// The hovered program's share of a total, on the header bars.
const HILITE: egui::Color32 = egui::Color32::from_rgb(222, 140, 51);
/// A dimmer `HILITE`, for the programs listed above the hovered one on the sorted column's header bar.
const FAINT_HILITE: egui::Color32 = egui::Color32::from_rgb(130, 88, 42);
/// A CPU core running hot, over 90%, on the per-core header graph.
const HOT: egui::Color32 = egui::Color32::from_rgb(170, 60, 55);
/// The per-core header bars, brighter than `BAR` so the thin bars on many-core CPUs stay legible.
const CORE_BAR: egui::Color32 = egui::Color32::from_rgb(90, 140, 210);
/// The names of the rows of memory that no program owns, like the kernel's pools.
const SYSTEM_TEXT: egui::Color32 = egui::Color32::from_rgb(130, 180, 245);
/// How far the shared-RAM figure sits left of the cell's right edge, so the "sh" numbers line up down the column.
const SH_INSET: f32 = 44.0;
/// The lines down a hovered column's left and right edges: a clear, transient hint it can be clicked to sort.
const HOVER_BORDER: egui::Color32 = egui::Color32::from_gray(150);
/// The subtle wash marking the sorted column: a gentle, lasting mark that doesn't distract.
const SORTED_COL: egui::Color32 = egui::Color32::from_rgba_premultiplied(13, 13, 13, 13);
const NAME_TIP: &str = "Programs by executable, and memory no program owns";
const RAM_TIP: &str = "Private RAM. The smaller \"sh\" figure is RAM shared with other programs, not counted in the main number";
const COMMIT_TIP: &str = "Private RAM + page file";
const CPU_TIP: &str = "Overall use";
const CORE_TIP: &str = "Max single-core use";
const VRAM_TIP: &str = "Video memory";
const DISK_TIP: &str = "Read + write speed";
const NET_TIP: &str = "Send + receive speed (needs admin)";
/// What the "10s" mark on a value column sorted by its recent peak means.
const PEAK_TIP: &str = "The \"10s\" mark means sorted by each program's highest value over the last 10 seconds (the faint bar); the number shows the current value";

/// What a process uses, or all the processes of a program.
#[derive(Default)]
pub struct Usage {
    /// Physical memory that only the process uses.
    pub ram: u64,
    /// Physical memory the process shares with others, like loaded DLLs and mapped files, on top of `ram`.
    pub shared: u64,
    /// Memory that only the process uses, also counting what was moved out of RAM.
    pub commit: u64,
    /// Dedicated video memory used on each GPU of `Sample::gpus`, missing ones being zero.
    pub vram: Vec<u64>,
    /// Share of the time of all CPU cores since the previous sample, in percent.
    pub cpu: f32,
    /// Busiest single thread as a share of one core, in percent, so a pegged core shows near 100.
    pub core: f32,
    /// Bytes per second read from and written to disk.
    pub disk: u64,
    /// Bytes per second sent and received over the network.
    pub net: u64,
}

impl Usage {
    fn add(&mut self, other: &Usage) {
        self.ram += other.ram;
        self.shared += other.shared;
        self.commit += other.commit;
        self.vram.resize(self.vram.len().max(other.vram.len()), 0);
        for (sum, vram) in self.vram.iter_mut().zip(&other.vram) {
            *sum += vram;
        }
        self.cpu += other.cpu;
        // A group's core use is its busiest process's, not the sum, so one pegged core stays at about 100.
        self.core = self.core.max(other.core);
        self.disk += other.disk;
        self.net += other.net;
    }
}

pub struct Proc {
    /// Zero for the rows of memory that no process owns, like the kernel's pools, which can't be ended.
    pub pid: u32,
    /// The process that started this one, for ending a whole tree.
    pub parent: u32,
    /// The executable's file name.
    pub name: String,
    /// What a row that isn't a process holds; empty for processes.
    pub tip: &'static str,
    pub usage: Usage,
}

/// How much of some memory is in use.
#[derive(Default)]
pub struct Total {
    pub used: u64,
    pub total: u64,
}

pub struct Gpu {
    pub name: String,
    pub vram: Total,
}

#[derive(Default)]
pub struct Sample {
    pub ram: Total,
    /// The commit charge against its limit.
    pub commit: Total,
    pub gpus: Vec<Gpu>,
    pub procs: Vec<Proc>,
    /// Bytes per second over all network interfaces.
    pub net: u64,
    /// Whether the processes carry their own network use, which needs admin rights.
    pub net_per_process: bool,
    /// The busy share of each CPU core, in percent.
    pub cores: Vec<f32>,
    pub details: Details,
}

/// Hardware facts for the column tooltips, empty until known.
#[derive(Default, Clone)]
pub struct Details {
    /// The CPU's name with its core and thread counts.
    pub cpu: String,
    /// The page file's size, and whether it's fixed or grows.
    pub pagefile: String,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
enum Col {
    Name,
    Ram,
    Commit,
    Vram(usize),
    Cpu,
    Core,
    Disk,
    Net,
}

impl Col {
    fn value(self, usage: &Usage) -> f64 {
        match self {
            Col::Name => 0.0,
            Col::Ram => usage.ram as f64,
            Col::Commit => usage.commit as f64,
            Col::Vram(gpu) => usage.vram.get(gpu).copied().unwrap_or(0) as f64,
            // As shown, for rows that show no CPU use to sort by name.
            Col::Cpu => (usage.cpu as f64 * 10.0).round() / 10.0,
            Col::Core => (usage.core as f64).round(),
            Col::Disk => usage.disk as f64,
            Col::Net => usage.net as f64,
        }
    }

    /// Nothing for zero, for the rest to stand out.
    fn text(self, usage: &Usage) -> String {
        let value = self.value(usage);
        match self {
            _ if value <= 0.0 => String::new(),
            Col::Cpu => format!("{value:.1}%"),
            Col::Core => format!("{value:.0}%"),
            Col::Disk | Col::Net => rate(value as u64),
            _ => size(value as u64),
        }
    }

    /// The system-wide use of the memory in the column.
    fn total(self, sample: &Sample) -> Option<&Total> {
        match self {
            Col::Ram => Some(&sample.ram),
            Col::Commit => Some(&sample.commit),
            Col::Vram(gpu) => sample.gpus.get(gpu).map(|gpu| &gpu.vram),
            _ => None,
        }
    }
}

/// The processes of a program, as those with the same executable.
struct Group<'a> {
    name: &'a str,
    usage: Usage,
    procs: Vec<&'a Proc>,
}

enum Row<'a> {
    Group(&'a Group<'a>),
    Proc(&'a Proc),
}

/// eframe's own record of the window, which it clamps onto a connected monitor and shrinks to fit the largest one on start.
/// The fields match eframe's internal `WindowSettings`, so we can keep the last normal place when quitting minimized or maximized.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct WindowPlace {
    inner_position_pixels: Option<egui::Pos2>,
    outer_position_pixels: Option<egui::Pos2>,
    fullscreen: bool,
    maximized: bool,
    inner_size_points: Option<egui::Vec2>,
}

struct App {
    rx: mpsc::Receiver<Sample>,
    sample: Sample,
    sort: Col,
    /// The second sort mode of the sorted column: Z to A for names, or by recent peak for values.
    alt: bool,
    filter: String,
    /// The programs with their processes listed below them.
    expanded: HashSet<String>,
    /// The program hovered last frame, whose contribution the header bars highlight.
    hovered: Option<String>,
    /// The recent per-program usage, newest last, for the background bars and the peak sort.
    history: VecDeque<HashMap<String, Usage>>,
    /// Whether to offer relaunching as admin, which per-process network needs.
    offer_admin: bool,
    /// Whether Ctrl+Shift+Esc currently opens this program instead of Task Manager.
    default_tm: bool,
    /// Whether the window has been confirmed on a monitor, so the off-screen check can stop.
    placed: bool,
    /// Each column's horizontal span from the last frame, to tell which column the pointer is over.
    col_ranges: Vec<egui::Rangef>,
    ctx: egui::Context,
    /// The last place the window was shown normally, kept across minimizing and maximizing.
    window: Option<WindowPlace>,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let (tx, rx) = mpsc::channel();
        let ctx = cc.egui_ctx.clone();
        // The bars, washes and near-white text all assume a dark background, so force it whatever the system theme is.
        ctx.set_theme(egui::ThemePreference::Dark);
        ctx.all_styles_mut(|style| {
            // Near-white text, brighter than egui's default gray, so the dense numbers read clearly.
            style.visuals.override_text_color = Some(egui::Color32::from_gray(235));
            // The header tooltips explain the columns, so show them at a quarter of egui's usual delay.
            style.interaction.tooltip_delay *= 0.25;
        });
        // Sampling takes a moment, which would stall the UI.
        std::thread::spawn({
            let ctx = ctx.clone();
            move || {
                let mut sampler = sampler::Sampler::new();
                loop {
                    let sample = sampler.sample();
                    if tx.send(sample).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                    // GPU and network setup is slow, so do it after the first sample so the core data shows at once.
                    sampler.prepare();
                    std::thread::sleep(REFRESH);
                }
            }
        });
        // Restore the last sorted column and its order, else the default of RAM by recent peak.
        let (sort, alt) = cc.storage.and_then(|storage| eframe::get_value::<(Col, bool)>(storage, SORT_KEY)).unwrap_or((Col::Ram, true));
        Self {
            rx,
            sample: Sample::default(),
            sort,
            alt,
            filter: String::new(),
            expanded: HashSet::new(),
            hovered: None,
            history: VecDeque::new(),
            offer_admin: sampler::offer_admin(),
            default_tm: sampler::is_default_task_manager(),
            placed: false,
            col_ranges: Vec::new(),
            window: cc.storage.and_then(|storage| eframe::get_value(storage, WINDOW_KEY)),
            ctx,
        }
    }

    /// The hover text for a column: what it means, any hardware detail, and the sort note.
    fn column_tip(&self, col: Col, tip: &str) -> String {
        let details = &self.sample.details;
        let detail = match col {
            Col::Cpu | Col::Core => details.cpu.as_str(),
            Col::Commit => details.pagefile.as_str(),
            Col::Vram(gpu) => self.sample.gpus.get(gpu).map_or("", |gpu| gpu.name.as_str()),
            _ => "",
        };
        let peak = if col == Col::Name { "" } else { PEAK_TIP };
        [tip, detail, peak].into_iter().filter(|line| !line.is_empty()).collect::<Vec<_>>().join("\n")
    }

    /// The largest value a column reached over the recent history, which the background bars scale against,
    /// so a brief spike doesn't rescale everything for just one frame.
    fn col_max(&self, col: Col) -> f64 {
        self.history.iter().flat_map(|snapshot| snapshot.values()).map(|usage| col.value(usage)).fold(0.0, f64::max)
    }

    /// A program's largest value in a column over the recent history, so a brief spike keeps it near the top.
    fn peak(&self, name: &str, col: Col) -> f64 {
        self.history.iter().filter_map(|snapshot| snapshot.get(name)).map(|usage| col.value(usage)).fold(0.0, f64::max)
    }

    /// Orders two programs by the sorted column: by name, by current value, or by recent peak in its alternate mode.
    fn order(&self, a: &Group, b: &Group) -> Ordering {
        if self.sort == Col::Name {
            let order = by_name(a.name, b.name);
            return if self.alt { order.reverse() } else { order };
        }
        let (a_value, b_value) = if self.alt {
            (self.peak(a.name, self.sort), self.peak(b.name, self.sort))
        } else {
            (self.sort.value(&a.usage), self.sort.value(&b.usage))
        };
        b_value.total_cmp(&a_value).then_with(|| by_name(a.name, b.name))
    }

    /// Only borrows the processes, for the filter to stay editable while the groups are shown.
    fn groups<'a>(&self, procs: &'a [Proc]) -> Vec<Group<'a>> {
        let filter = self.filter.to_lowercase();
        let mut groups = HashMap::new();
        for proc in procs.iter().filter(|proc| proc.name.to_lowercase().contains(&filter)) {
            let group = groups.entry(proc.name.as_str()).or_insert_with(|| Group { name: &proc.name, usage: Usage::default(), procs: Vec::new() });
            group.usage.add(&proc.usage);
            group.procs.push(proc);
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        for group in &mut groups {
            // Within a program, always by the current value, since peaks are tracked per program only.
            let order = |a: &&Proc, b: &&Proc| match self.sort {
                Col::Name => Ordering::Equal,
                col => col.value(&b.usage).total_cmp(&col.value(&a.usage)),
            };
            group.procs.sort_by(|a, b| order(a, b).then(a.pid.cmp(&b.pid)));
        }
        groups.sort_by(|a, b| self.order(a, b));
        groups
    }

    fn table_ui(&mut self, ui: &mut egui::Ui) {
        let gpus = &self.sample.gpus;
        let mut cols = vec![
            (Col::Name, "Name".to_owned(), NAME_TIP.to_owned()),
            (Col::Cpu, "CPU".to_owned(), CPU_TIP.to_owned()),
            (Col::Core, "Core".to_owned(), CORE_TIP.to_owned()),
            (Col::Ram, "RAM".to_owned(), RAM_TIP.to_owned()),
            (Col::Commit, "Commit".to_owned(), COMMIT_TIP.to_owned()),
        ];
        for i in 0..gpus.len() {
            cols.push((Col::Vram(i), vram_title(gpus, i), VRAM_TIP.to_owned()));
        }
        cols.push((Col::Disk, "Disk".to_owned(), DISK_TIP.to_owned()));
        cols.push((Col::Net, "Network".to_owned(), NET_TIP.to_owned()));
        // The biggest each column reached recently, to scale its background bars against.
        // Core scales against one whole core instead, so a near-full bar means a pegged core.
        let col_max: Vec<f64> = cols.iter().map(|(col, ..)| if *col == Col::Core { 100.0 } else { self.col_max(*col) }).collect();

        let groups = self.groups(&self.sample.procs);
        let mut rows = Vec::new();
        for group in &groups {
            rows.push(Row::Group(group));
            if group.procs.len() > 1 && self.expanded.contains(group.name) {
                rows.extend(group.procs.iter().map(|&proc| Row::Proc(proc)));
            }
        }
        let cpu: f32 = self.sample.procs.iter().map(|proc| proc.usage.cpu).sum();
        let disk: u64 = self.sample.procs.iter().map(|proc| proc.usage.disk).sum();
        // The hovered program's usage, to highlight its share on each header bar.
        let hovered_usage = self.hovered.as_deref().and_then(|name| groups.iter().find(|group| group.name == name)).map(|group| &group.usage);
        // The sorted column's sum over the programs listed above the hovered one, so its header shows the top programs' running total.
        let above: f64 = match (hovered_usage, self.hovered.as_deref()) {
            (Some(_), Some(name)) => groups.iter().take_while(|group| group.name != name).map(|group| self.sort.value(&group.usage)).sum(),
            _ => 0.0,
        };

        let (mut sort, mut toggle, mut relaunch, mut kill, mut now_hovered) = (None, None, false, None, None);
        // Set when the top-row buttons are clicked: launch Task Manager, or toggle whether we replace it.
        let (mut launch_tm, mut set_default) = (false, None);
        let height = ui.spacing().interact_size.y;
        // The totals or filter on top, the sortable title below.
        let header_height = 2.0 * (height + ui.spacing().item_spacing.y);
        let right = egui::Layout::right_to_left(egui::Align::Center);
        // Selectable text would take the clicks on the rows.
        ui.style_mut().interaction.selectable_labels = false;
        // A regular, always-present scroll bar rather than a thin overlay one.
        ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
        // The table's horizontal span, to detect hovering a whole row, and its vertical span, to draw the hovered column's edges.
        let table_x = ui.max_rect().x_range();
        let table_y = ui.max_rect().y_range();
        // Each column's explanation, and which column the pointer is over, using last frame's spans since this frame's come from rendering.
        let col_tips: Vec<String> = cols.iter().map(|(col, _, tip)| self.column_tip(*col, tip)).collect();
        let col_hover = ui
            .ctx()
            .pointer_latest_pos()
            .filter(|pos| ui.max_rect().contains(*pos))
            .and_then(|pos| self.col_ranges.iter().position(|range| range.contains(pos.x)));
        let mut col_ranges = vec![egui::Rangef::new(0.0, 0.0); cols.len()];
        let mut builder = TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .sense(egui::Sense::click())
            .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center));
        // Only a remainder column that isn't resizable keeps filling the table as the window resizes.
        builder = builder.column(Column::remainder().at_least(150.0).clip(true).resizable(false));
        // The middle columns, with RAM wider to fit its aligned "sh" sub-column beside the main figure.
        for (col, ..) in &cols[1..cols.len() - 1] {
            let width = if *col == Col::Ram { 105.0 } else { 95.0 };
            builder = builder.column(Column::initial(width).at_least(50.0));
        }
        // The last column isn't resizable, so its right edge can't be grabbed by mistake instead of the scroll bar.
        builder = builder.column(Column::initial(95.0).at_least(50.0).resizable(false));
        builder
            .header(header_height, |mut header| {
                for (i, (col, title, _)) in cols.iter().enumerate() {
                    header.col(|ui| {
                        // Remember this column's span for next frame's pointer test, and wash the sorted and hovered columns.
                        col_ranges[i] = ui.max_rect().expand2(0.5 * ui.spacing().item_spacing).x_range();
                        paint_col_tint(ui, self.sort == *col);
                        // A click anywhere in the header cell sorts, and hovering it explains the column; added first so the search box on top still gets its own clicks.
                        let cell = ui.interact(ui.max_rect(), ui.id().with("sort"), egui::Sense::click()).on_hover_text(col_tips[i].clone());
                        let align = if *col == Col::Name { egui::Align::Min } else { egui::Align::Max };
                        // The running total above the hovered program, as a fraction of a header bar's full width; only on the sorted column.
                        let sorted = self.sort == *col;
                        let faint = |full: f64| if sorted { (above / full.max(1.0)) as f32 } else { 0.0 };
                        ui.with_layout(egui::Layout::top_down(align), |ui| {
                            // The totals, or the filter, above the title.
                            match col {
                                Col::Name => {
                                    ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Search").desired_width(ui.available_width()));
                                }
                                Col::Cpu => {
                                    let contrib = hovered_usage.map(|usage| usage.cpu / 100.0).unwrap_or(0.0);
                                    total_bar(ui, cpu / 100.0, faint(100.0), contrib, format!("{cpu:.0}%"));
                                }
                                // A small bar per core, the busiest core's figure on top.
                                Col::Core => {
                                    let cores = &self.sample.cores;
                                    let max = cores.iter().copied().fold(0.0_f32, f32::max);
                                    let text = if cores.is_empty() { String::new() } else { format!("{max:.0}%") };
                                    core_bars(ui, cores, text);
                                }
                                // Rates have no capacity, so the bar (the current total split by program) only appears while hovering.
                                Col::Disk => {
                                    let contrib = hovered_usage.map(|usage| usage.disk as f32 / disk.max(1) as f32).unwrap_or(0.0);
                                    total_bar(ui, if hovered_usage.is_some() { 1.0 } else { 0.0 }, faint(disk as f64), contrib, rate(disk));
                                }
                                Col::Net => {
                                    let contrib = hovered_usage.map(|usage| usage.net as f32 / self.sample.net.max(1) as f32).unwrap_or(0.0);
                                    total_bar(ui, if hovered_usage.is_some() { 1.0 } else { 0.0 }, faint(self.sample.net as f64), contrib, rate(self.sample.net));
                                }
                                _ => match col.total(&self.sample).filter(|total| total.total > 0) {
                                    Some(total) => {
                                        // Windows counts memory that programs share, like GPU surfaces, in each of them, so their sum can top
                                        // what's really used; the highlights shrink to fit so the whole list lands on the used total.
                                        let listed: f64 = self.sample.procs.iter().map(|proc| col.value(&proc.usage)).sum();
                                        let fit = if listed > total.used as f64 { total.used as f64 / listed } else { 1.0 };
                                        let contrib = hovered_usage.map(|usage| (col.value(usage) * fit) as f32 / total.total as f32).unwrap_or(0.0);
                                        total_bar(ui, total.used as f32 / total.total as f32, faint(total.total as f64 / fit), contrib, used_of(total));
                                    }
                                    None => total_bar(ui, 0.0, 0.0, 0.0, String::new()),
                                },
                            }
                            // Plain text, not a button: the whole header cell handles the click to sort and the tooltip.
                            let arrow = sort_arrow(self.sort == *col, *col, self.alt);
                            if *col == Col::Name {
                                // The launch and replace buttons sit in the empty space beside the "Name" title, costing no extra height.
                                egui::containers::Sides::new().show(
                                    ui,
                                    |ui| {
                                        ui.label(format!("{title}{arrow}"));
                                    },
                                    |ui| {
                                        if ui.button("Task Mgr").on_hover_text("Open the built-in Windows Task Manager").clicked() {
                                            launch_tm = true;
                                        }
                                        let (label, tip) = if self.default_tm {
                                            ("Default \u{2713}", "tuskManager opens on Ctrl+Shift+Esc; click to restore Windows Task Manager (needs admin)")
                                        } else {
                                            ("Set default", "Open tuskManager on Ctrl+Shift+Esc instead of Task Manager (needs admin)")
                                        };
                                        if ui.button(label).on_hover_text(tip).clicked() {
                                            set_default = Some(!self.default_tm);
                                        }
                                    },
                                );
                            } else {
                                ui.label(format!("{title}{arrow}"));
                            }
                        });
                        if cell.clicked() {
                            sort = Some(*col);
                        }
                    });
                }
            })
            .body(|body| {
                body.rows(height, rows.len(), |mut row| {
                    let index = row.index();
                    let (usage, group) = match rows[index] {
                        Row::Group(group) => (&group.usage, Some(group)),
                        Row::Proc(proc) => (&proc.usage, None),
                    };
                    let mut acted = false;
                    // Whether the pointer is anywhere in this row's full-width band; shared so every cell highlights, not just the name.
                    // Asked of the context, since a cell's own check clips to the cell; only the row's visible height counts.
                    let mut row_hovered = false;
                    row.col(|ui| {
                        let band = egui::Rect::from_x_y_ranges(table_x, ui.clip_rect().y_range());
                        row_hovered = ui.ctx().rect_contains_pointer(ui.layer_id(), band);
                        if row_hovered {
                            paint_row_hover(ui);
                        }
                        paint_col_tint(ui, self.sort == Col::Name);
                        if row_hovered {
                            now_hovered = Some(match rows[index] {
                                Row::Group(group) => group.name.to_owned(),
                                Row::Proc(proc) => proc.name.clone(),
                            });
                        }
                        egui::containers::Sides::new().shrink_left().show(
                            ui,
                            |ui| match rows[index] {
                                Row::Group(group) => {
                                    let several = group.procs.len() > 1;
                                    let arrow = match (several, self.expanded.contains(group.name)) {
                                        (false, _) => "",
                                        (true, false) => "⏵",
                                        (true, true) => "⏷",
                                    };
                                    ui.add_sized([12.0, height], egui::Label::new(arrow));
                                    let text = if several { format!("{} ({})", group.name, group.procs.len()) } else { group.name.to_owned() };
                                    // Memory that no process owns, in blue with what it holds on hover.
                                    let tip = group.procs[0].tip;
                                    if tip.is_empty() {
                                        ui.add(egui::Label::new(text).truncate());
                                    } else {
                                        ui.add(egui::Label::new(egui::RichText::new(text).color(SYSTEM_TEXT)).truncate()).on_hover_text(tip);
                                    }
                                }
                                Row::Proc(proc) => {
                                    ui.add_space(32.0);
                                    ui.label(egui::RichText::new(format!("PID {}", proc.pid)).weak());
                                }
                            },
                            |ui| {
                                if !row_hovered {
                                    return;
                                }
                                let (pids, what) = match rows[index] {
                                    Row::Group(group) => (group.procs.iter().map(|proc| proc.pid).collect::<Vec<_>>(), "program"),
                                    Row::Proc(proc) => (vec![proc.pid], "process"),
                                };
                                // Memory rows that aren't processes can't be ended.
                                if pids.contains(&0) {
                                    return;
                                }
                                if ui.button("Kill").on_hover_text(format!("End this {what} and everything it started")).clicked() {
                                    kill = Some(with_descendants(&self.sample.procs, &pids));
                                    acted = true;
                                }
                                if ui.button("Close").on_hover_text(format!("End this {what}")).clicked() {
                                    kill = Some(pids);
                                    acted = true;
                                }
                            },
                        );
                    });
                    for (i, (col, ..)) in cols.iter().enumerate().skip(1) {
                        row.col(|ui| {
                            if row_hovered {
                                paint_row_hover(ui);
                            }
                            paint_col_tint(ui, self.sort == *col);
                            // The Network column is empty without admin rights, so the prompt to get them goes there.
                            if *col == Col::Net && self.offer_admin {
                                admin_prompt(ui, index, &mut relaunch);
                                return;
                            }
                            if col_max[i] > 0.0 {
                                let rect = ui.max_rect();
                                let current = col.value(usage);
                                // A faint bar for the recent peak behind the solid bar for the current value; only programs keep a peak.
                                let peak = group.map(|group| self.peak(group.name, *col)).unwrap_or(current).max(current);
                                paint_bar(ui, rect, (peak / col_max[i]) as f32, FAINT);
                                paint_bar(ui, rect, (current / col_max[i]) as f32, BAR);
                            }
                            // The main value, the same way in every column so the figures line up across columns.
                            ui.with_layout(right, |ui| ui.label(col.text(usage)));
                            // The shared-RAM figure, painted at a fixed inset so these line up down their own sub-column.
                            if *col == Col::Ram && usage.shared > 0 {
                                let rect = ui.max_rect();
                                let at = egui::pos2(rect.right() - ui.spacing().item_spacing.x - SH_INSET, rect.center().y);
                                ui.painter().text(at, egui::Align2::RIGHT_CENTER, format!("{} sh", size(usage.shared)), egui::TextStyle::Small.resolve(ui.style()), ui.visuals().text_color());
                            }
                            // A click anywhere in a value column sorts by it; its tooltip shows here too, not just in the header.
                            let cell = ui.interact(ui.max_rect(), ui.id().with("sort"), egui::Sense::click()).on_hover_text(col_tips[i].clone());
                            if cell.clicked() {
                                sort = Some(*col);
                            }
                        });
                    }
                    // Clicking a value cell sorts (handled above); only a click landing on the name row toggles its processes.
                    if let Some(group) = group
                        && group.procs.len() > 1
                        && !acted
                        && row.response().clicked()
                    {
                        toggle = Some(group.name.to_owned());
                    }
                });
            });

        self.col_ranges = col_ranges;
        // Outline the hovered column down its left and right edges, on top of the whole table so nothing hides the lines.
        if let Some(range) = col_hover.and_then(|i| self.col_ranges.get(i).copied()) {
            let stroke = egui::Stroke::new(1.0, HOVER_BORDER);
            ui.painter().vline(range.min, table_y, stroke);
            ui.painter().vline(range.max, table_y, stroke);
        }
        self.hovered = now_hovered;
        if let Some(col) = sort {
            // A new column starts by its recent peak (names A to Z); clicking it again flips to the current value (names Z to A).
            self.alt = if col == self.sort { !self.alt } else { col != Col::Name };
            self.sort = col;
        }
        if let Some(name) = toggle
            && !self.expanded.remove(&name)
        {
            self.expanded.insert(name);
        }
        if relaunch {
            sampler::relaunch_as_admin();
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if let Some(pids) = kill {
            sampler::end(&pids);
        }
        if launch_tm {
            sampler::launch_task_manager();
        }
        if let Some(enable) = set_default {
            sampler::set_default_task_manager(enable);
            // The elevated edit runs in the background; assume it took, and re-check it each sample.
            self.default_tm = enable;
        }
    }
}

impl eframe::App for App {
    /// Only the window place is kept, in window.ron, not the egui memory.
    fn persist_egui_memory(&self) -> bool {
        false
    }

    /// A window minimized at exit has no size and sits off screen, and a maximized one is the size of the screen,
    /// so both keep the place from the last time the window was shown normally.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let minimized = self.ctx.input(|i| i.viewport().minimized == Some(true));
        if let Some(now) = eframe::get_value::<WindowPlace>(storage, WINDOW_KEY).filter(|_| !minimized) {
            self.window = Some(match self.window {
                Some(normal) if now.maximized || now.fullscreen => WindowPlace { maximized: now.maximized, fullscreen: now.fullscreen, ..normal },
                _ => now,
            });
        }
        match &self.window {
            Some(window) => eframe::set_value(storage, WINDOW_KEY, window),
            // Never shown normally yet: saved unreadable, for the next start to use the default window.
            None => storage.set_string(WINDOW_KEY, String::new()),
        }
        eframe::set_value(storage, SORT_KEY, &(self.sort, self.alt));
    }

    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // eframe misplaces saved positions across mixed-DPI monitors and can restore onto a gone or VR display, so place it ourselves.
        if !self.placed {
            let saved = self.window.and_then(|place| place.outer_position_pixels).map(|pos| (pos.x as i32, pos.y as i32));
            self.placed = sampler::place_window(saved);
        }
        if let Some(sample) = self.rx.try_iter().last() {
            self.sample = sample;
            // Cheap, and keeps the button right if the setting changed elsewhere or a prompt was cancelled.
            self.default_tm = sampler::is_default_task_manager();
            let mut snapshot: HashMap<String, Usage> = HashMap::new();
            for proc in &self.sample.procs {
                snapshot.entry(proc.name.clone()).or_default().add(&proc.usage);
            }
            self.history.push_back(snapshot);
            while self.history.len() > HISTORY {
                self.history.pop_front();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| self.table_ui(ui));
    }
}

/// Ignoring case, unless that's the only difference.
fn by_name(a: &str, b: &str) -> Ordering {
    a.to_lowercase().cmp(&b.to_lowercase()).then_with(|| a.cmp(b))
}

/// The given processes plus every process descended from them, by following parent links.
fn with_descendants(procs: &[Proc], roots: &[u32]) -> Vec<u32> {
    let mut all: HashSet<u32> = roots.iter().copied().collect();
    loop {
        let before = all.len();
        for proc in procs {
            if all.contains(&proc.parent) {
                all.insert(proc.pid);
            }
        }
        if all.len() == before {
            return all.into_iter().collect();
        }
    }
}

/// Like "0.3 MB", "512 MB" or "3.2 GB".
fn size(bytes: u64) -> String {
    let (unit, decimals, name) = unit(bytes);
    format!("{:.*} {name}", decimals, bytes as f64 / unit)
}

/// Like "18.2 / 31.2 GB", both in the unit that suits the total.
fn used_of(total: &Total) -> String {
    let (unit, decimals, name) = unit(total.total);
    format!("{:.*} / {:.*} {name}", decimals, total.used as f64 / unit, decimals, total.total as f64 / unit)
}

/// The unit that suits a size, how many decimals to show it with, and the unit's name.
fn unit(bytes: u64) -> (f64, usize, &'static str) {
    const MB: f64 = (1 << 20) as f64;
    match bytes as f64 / MB {
        mb if mb < 10.0 => (MB, 1, "MB"),
        mb if mb < 1000.0 => (MB, 0, "MB"),
        _ => (MB * 1024.0, 1, "GB"),
    }
}

/// A transfer speed like "340 KB/s" or "12 MB/s", down to the kilobyte since idle rates are small.
fn rate(bytes_per_sec: u64) -> String {
    let kb = bytes_per_sec as f64 / 1024.0;
    if kb < 1.0 {
        String::new()
    } else if kb < 1000.0 {
        format!("{kb:.0} KB/s")
    } else if kb < 1000.0 * 1024.0 {
        format!("{:.1} MB/s", kb / 1024.0)
    } else {
        format!("{:.1} GB/s", kb / 1024.0 / 1024.0)
    }
}

/// "VRAM", followed by the GPU's vendor when there are several, like on laptops with integrated graphics.
fn vram_title(gpus: &[Gpu], gpu: usize) -> String {
    if gpus.len() == 1 {
        return "VRAM".to_owned();
    }
    let vendor = gpus[gpu].name.split_whitespace().next().unwrap_or_default();
    format!("VRAM {}", vendor.replace("(R)", "").replace("(TM)", ""))
}

/// One centered, dim line of the Network-column admin prompt.
fn prompt_line(ui: &mut egui::Ui, text: &str) {
    ui.vertical_centered(|ui| ui.label(egui::RichText::new(text).weak()));
}

/// The admin prompt spread down the Network cells, which are empty without admin rights, with the button to relaunch elevated.
fn admin_prompt(ui: &mut egui::Ui, row: usize, relaunch: &mut bool) {
    match row {
        // A couple of empty cells at the top, then the message split across cells, then the button.
        2 => prompt_line(ui, "Admin"),
        3 => prompt_line(ui, "needed for"),
        4 => prompt_line(ui, "network"),
        5 => {
            ui.vertical_centered(|ui| {
                if ui.button("Run as admin").on_hover_text("Relaunch with admin rights to show each program's network use").clicked() {
                    *relaunch = true;
                }
            });
        }
        _ => {}
    }
}

/// Fills a cell with the theme's row-hover color, so hovering anywhere in a row highlights the whole row.
fn paint_row_hover(ui: &egui::Ui) {
    let rect = ui.max_rect().expand2(0.5 * ui.spacing().item_spacing);
    ui.painter().rect_filled(rect, egui::CornerRadius::ZERO, ui.visuals().widgets.hovered.bg_fill);
}

/// Washes the sorted column, out to the column dividers, behind its content.
fn paint_col_tint(ui: &egui::Ui, selected: bool) {
    if selected {
        let rect = ui.max_rect().expand2(0.5 * ui.spacing().item_spacing);
        ui.painter().rect_filled(rect, egui::CornerRadius::ZERO, SORTED_COL);
    }
}

/// Fills the left part of a rectangle, flat, as a background bar behind a value.
fn paint_bar(ui: &egui::Ui, rect: egui::Rect, fraction: f32, color: egui::Color32) {
    paint_span(ui, rect, 0.0, fraction, color);
}

/// Fills the part of a rectangle between two fractions of its width, flat.
fn paint_span(ui: &egui::Ui, rect: egui::Rect, from: f32, to: f32, color: egui::Color32) {
    let (from, to) = (from.clamp(0.0, 1.0), to.clamp(0.0, 1.0));
    if from < to {
        let span = egui::Rect::from_x_y_ranges(rect.left() + rect.width() * from..=rect.left() + rect.width() * to, rect.y_range());
        ui.painter().rect_filled(span, egui::CornerRadius::ZERO, color);
    }
}

/// A full-width total: a flat background bar with the hovered program's share highlighted, and the figure right-aligned on top.
/// `above` is the faint share of the programs listed above the hovered one, which its own share then continues from.
fn total_bar(ui: &mut egui::Ui, used: f32, above: f32, contrib: f32, text: String) {
    let height = ui.spacing().interact_size.y;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    paint_bar(ui, rect, used, BAR);
    paint_span(ui, rect, 0.0, above, FAINT_HILITE);
    paint_span(ui, rect, above, above + contrib, HILITE);
    let font = egui::TextStyle::Button.resolve(ui.style());
    let at = rect.right_center() - egui::vec2(ui.spacing().item_spacing.x, 0.0);
    ui.painter().text(at, egui::Align2::RIGHT_CENTER, text, font, ui.visuals().text_color());
}

/// A small vertical bar per CPU core across the header, hot cores in red, with the busiest core's figure on the right.
fn core_bars(ui: &mut egui::Ui, cores: &[f32], text: String) {
    let height = ui.spacing().interact_size.y;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    let ppp = ui.ctx().pixels_per_point();
    // Snap the shared baseline to a physical pixel so every bar starts on the exact same line.
    let bottom = rect.bottom().round_to_pixels(ppp);
    // Bars use the whole column width; the figure is drawn over them, not beside them.
    let step = rect.width() / cores.len().max(1) as f32;
    for (i, &usage) in cores.iter().enumerate() {
        let fraction = (usage / 100.0).clamp(0.0, 1.0);
        // Bars touch edge to edge, snapped to pixels, so a dense many-core CPU reads clearly and evenly.
        let x0 = (rect.left() + i as f32 * step).round_to_pixels(ppp);
        let x1 = (rect.left() + (i + 1) as f32 * step).round_to_pixels(ppp);
        let top = (rect.bottom() - rect.height() * fraction).round_to_pixels(ppp);
        let bar = egui::Rect::from_min_max(egui::pos2(x0, top), egui::pos2(x1, bottom));
        // Every core over 90% turns red, each judged on its own.
        ui.painter().rect_filled(bar, egui::CornerRadius::ZERO, if usage > 90.0 { HOT } else { CORE_BAR });
    }
    // The busiest-core figure, right-aligned on top of the bars.
    let font = egui::TextStyle::Button.resolve(ui.style());
    let color = ui.visuals().text_color();
    let gap = ui.spacing().item_spacing.x;
    let galley = ui.painter().layout_no_wrap(text, font, color);
    ui.painter().galley(egui::pos2(rect.right() - gap - galley.size().x, rect.center().y - galley.size().y / 2.0), galley, color);
}

/// The marker on the sorted column's title: up for names A to Z, down otherwise, with the recent-peak mode noted.
fn sort_arrow(sorted: bool, col: Col, alt: bool) -> &'static str {
    match (sorted, col, alt) {
        (false, ..) => "",
        (true, Col::Name, false) => " ⏶",
        (true, Col::Name, true) => " ⏷",
        (true, _, false) => " ⏷",
        (true, _, true) => " ⏷10s",
    }
}

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)).unwrap_or_default()
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        // Releases are plain numbers, but Cargo needs x.y.z, so only the major part is shown.
        viewport: egui::ViewportBuilder::default().with_title(concat!("tuskManager v", env!("CARGO_PKG_VERSION_MAJOR"))).with_inner_size([1050.0, 450.0]),
        // Without a path eframe keeps its window record in the user's app data; here it sits next to the exe.
        persistence_path: Some(exe_dir().join("data").join("window.ron")),
        ..Default::default()
    };
    eframe::run_native("tuskManager", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_own_process() {
        let mut sampler = sampler::Sampler::new();
        sampler.sample();
        std::thread::sleep(Duration::from_millis(300));
        let sample = sampler.sample();
        let own = sample.procs.iter().find(|proc| proc.pid == std::process::id()).expect("own process listed");
        assert!(own.usage.ram > 0 && !own.name.is_empty());
        assert!(0 < sample.ram.used && sample.ram.used <= sample.ram.total);
        assert!(sample.commit.used <= sample.commit.total);
        let cpu: f32 = sample.procs.iter().map(|proc| proc.usage.cpu).sum();
        assert!(cpu <= 101.0, "total CPU {cpu}%");
    }
}
