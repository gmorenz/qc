#![allow(incomplete_features)]
#![feature(
    format_args_capture,
    const_generics,
    const_evaluatable_checked,
)]

use std::rc::Rc;
use std::str::FromStr;
use std::f64::consts::TAU;

use druid::{
    *,
    widget::*,
    piet::kurbo::Circle,
};

use im::Vector;

use qc::*;

#[derive(Clone, Data, Lens)]
struct ChessGame {
    state: Rc<ChessState>,
    history: Vector<HistItem>,
    selection: Selection,

    // Derivative values that are worth caching
    // TODO: Worth putting behind an rc?
    #[data(ignore)]
    occupancy_prob: Vec<f32>,
    #[data(ignore)]
    occupancy_prob_relative: Option<Vec<f32>>,

    // More or less constant values that need loading
    piece_svgs: Rc<[SvgData; 12]>,
}

#[derive(Copy, Clone, Data, Lens, Debug, PartialEq)]
struct HistItem {
    #[data(same_fn = "PartialEq::eq")]
    chess_move: Move,
    measure: Option<bool>,
}

impl HistItem {
    fn to_string(self) -> String {
        match self.measure {
            Some(true) => format!("{:.5}.m1", self.chess_move),
            Some(false) => format!("{:.5}.m0", self.chess_move),
            None => format!("{:.5}  ", self.chess_move),
        }
    }
}

fn main() {
    let seed = u64::from_str(&std::env::args().nth(1).unwrap()).unwrap();
    unsafe {
        extern "C" { fn srand48(x: u64); }
        srand48(seed);
    }

    let svg = |src| SvgData::from_str(src).unwrap();
    let piece_svgs = Rc::new([
        svg(include_str!("Chess_plt45.svg")),
        svg(include_str!("Chess_nlt45.svg")),
        svg(include_str!("Chess_blt45.svg")),
        svg(include_str!("Chess_rlt45.svg")),
        svg(include_str!("Chess_klt45.svg")),
        svg(include_str!("Chess_qlt45.svg")),
        svg(include_str!("Chess_pdt45.svg")),
        svg(include_str!("Chess_ndt45.svg")),
        svg(include_str!("Chess_bdt45.svg")),
        svg(include_str!("Chess_rdt45.svg")),
        svg(include_str!("Chess_kdt45.svg")),
        svg(include_str!("Chess_qdt45.svg")),
    ]);

    let main_window = WindowDesc::new(move || build_chess_widget())
        // Convenient for building with cargo watch -x run to make it spawn
        // in an out of the way location :)
        .window_size((1000.0, 550.0))
        .set_position((2000.0, 1400.0))
    ;

    let state = ChessState::new_game();
    let occupancy_prob = get_probs(&state);


    let game = ChessGame {
        state: Rc::new(state),
        history: Vector::new(),
        selection: Selection::None,

        occupancy_prob,
        occupancy_prob_relative: None,

        piece_svgs,
    };

    AppLauncher::with_window(main_window)
        .launch(game)
        .expect("Failed to launch");
}

fn build_chess_widget() -> impl Widget<ChessGame> {
    // Start board row

    let board = QChessBoard::new().center()
        .padding(SQUARE_SIZE / 2.0);

    fn build_hist_item() -> impl Widget<(usize, HistItem, Option<HistItem>)> {
        let label = Label::new(|&(i, _, _): &(usize, HistItem, Option<HistItem>), _: &Env|
            format!("{i:.2}."));
        let white_move = Label::new(|&(_, m, _): &(usize, HistItem, Option<HistItem>), _: &Env|
            m.to_string());
        let black_move = Label::new(|&(_, _, m): &(usize, HistItem, Option<HistItem>), _: &Env|
            if let Some(m) = m {
                m.to_string()
            } else { format!("") }
        );

        Flex::row()
            .with_child(label.fix_width(30.0))
            .with_child(white_move.fix_width(80.0))
            .with_child(black_move.fix_width(80.0))
    }

    fn map_hist(history: &Vector<HistItem>) -> Vector<(usize, HistItem, Option<HistItem>)> {
        (0.. (history.len() + 1)/2)
            .into_iter()
            .map(|i| (i, history[i * 2], history.get(i * 2 + 1).map(|&x| x)))
            .collect()
    }
    let move_hist = List::new(build_hist_item)
        .lens(lens::Map::new(
            map_hist,
            |data, new| debug_assert_eq!(map_hist(&*data), new, "Move numbers is immutable")
        ))
        .lens(ChessGame::history);

    let move_hist_scroll = Scroll::new(move_hist)
        .vertical()
        .stick_to_bottom(true)
        .fix_size(30.0 + 80.0 + 80.0, SQUARE_SIZE * 6.0)
        .padding(SQUARE_SIZE / 2.0);

    // TODO: Would be nice to give this a min-size of 1 piece...
    let captured_pieces = Flex::column()
        .with_child(QCaptured{white_pieces: true})
        .with_child(QCaptured{white_pieces: false});

    let board_row = Flex::row()
        .with_child(board)
        .with_child(captured_pieces)
        .with_child(move_hist_scroll);

    // Start state list

    // TODO: Create and use an "infinite scroll list"
    fn map_qchess_state(game: &ChessGame) -> Vector<QChessStateData> {
        fn basis_vec<const N: usize>(state: &QuantumState<N>, critera: Option<usize>, limit: usize) -> Vec<(BasisVector<64>, Complex)>
        where [u8; (N+7)/8]: {
            state.iter()
                .filter(|(bv, _)| critera.map(|c| bv[c]).unwrap_or(true))
                .take(limit)
                .map(|(bv, val)| (bv.resize(), val))
                .collect()
        }

        let critera = game.selection.get_filter_qbit();

        let bvs = apply_monomorphized_fn!(&game.state.occupancy.state, basis_vec, critera, 256);
        bvs.into_iter().map(|(bv, val)| QChessStateData {
            svgs: game.piece_svgs.clone(),
            pieces: game.state.pieces,
            occupancy: bv,
            val
        }).collect()
    }

    let state_list_list = List::new(|| QChessStateBoard {})
        .horizontal()
        .with_spacing(25.0)
        .lens(lens::Map::new(
            map_qchess_state,
            |_data, _new| (),
        ));

    let state_list = Scroll::new(state_list_list)
        .horizontal()
        .padding(50.0);

    Flex::column()
        .with_child(board_row.align_left())
        .with_child(state_list)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Selection {
    None,
    Start(Square),
    StartMerge(Square),
    MiddleMerge(Square, Square),
    StartSplit(Square, Square),
}

impl Data for Selection {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

impl Selection {
    fn left_click(&mut self, state: &ChessState, sq: Square) -> Option<Move> {
        use Selection::*;
        match *self {
            None => {
                *self = Start(sq);
                Option::None
            },
            Start(from) => {
                *self = None;
                // TODO: Add a new selection state for promotion with a selection ui...
                if sq.row.perspective(state.is_white_turn) == Row::R8
                && state.pieces[from.get_idx()].map(|p| p.kind == PieceKind::Pawn).unwrap_or(false) {
                    Some(Move::Promote{from, to: sq, kind: PieceKind::Queen})
                } else {
                    Some(Move::Simple{from, to: sq})
                }
            },
            StartMerge(from_1) => {
                *self = MiddleMerge(from_1, sq);
                Option::None
            },
            MiddleMerge(from_1, from_2) => {
                *self = None;
                Some(Move::Merge{from_1, from_2, to: sq})
            },
            StartSplit(from, to_1) => {
                *self = None;
                Some(Move::Split{from, to_1, to_2: sq})
            }
        }
    }

    fn right_click(&mut self, sq: Square) {
        use Selection::*;
        match *self {
            None => {
                *self = StartMerge(sq)
            },
            Start(from) => {
                *self = StartSplit(from, sq)
            }
            StartMerge(..) | MiddleMerge(..) | StartSplit(..) => {
                *self = None
            }
        }
    }
}

impl Selection {
    fn get_filter_qbit(self) -> Option<usize> {
        match self {
            Selection::Start(from) | Selection::StartMerge(from) => {
                Some(from.get_idx())
            },
            _ => None
        }
    }
}

impl ChessGame {
    fn set_occupancy_prob_rel(&mut self) {
        // TODO: Other ways to select squares?
        // Combining with merge move is weird...
        match self.selection.get_filter_qbit() {
            Some(from)
                => self.occupancy_prob_relative = {
                    let prob = self.occupancy_prob[from];
                    if prob > 0.999 { None }
                    else {
                        let mut probs = get_probs_given(&self.state, &[from]);
                        probs.iter_mut().for_each(|p| *p /= prob);
                        Some(probs)
                    }
                },
            None => self.occupancy_prob_relative = None,
        };
    }

    fn svg(&self, piece: Piece) -> &SvgData {
        &self.piece_svgs[svg_idx(piece)]
    }
}

fn svg_idx(piece: Piece) -> usize {
    let idx_piece = match piece.kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::King => 4,
        PieceKind::Queen => 5,
    };
    idx_piece + if piece.is_white { 0 } else { 6 }
}

struct QChessBoard {
}

impl QChessBoard {
    fn new() -> Self {
        QChessBoard {}
    }
}

const SQUARE_SIZE: f64 = 80.0;
const CHESS_BOARD_SIZE: Size = Size::new(SQUARE_SIZE * 8.0, SQUARE_SIZE * 8.0);

const WHITE_SQUARE_COLOR: Color = Color::rgb8(200, 200, 255);
const BLACK_SQUARE_COLOR: Color = Color::rgb8(40, 120, 40);

fn origin(mut i: usize, j: usize) -> Point {
    i = 7 - i;
    Point::new(j as f64 * SQUARE_SIZE, i as f64 * SQUARE_SIZE)
}

fn sq_rect(i: usize, j: usize) -> Rect {
    let sq_size = Size::new(SQUARE_SIZE, SQUARE_SIZE);
    Rect::from_origin_size(origin(i, j), sq_size)
}

fn draw_board(ctx: &mut PaintCtx) {
    // Draw board
    ctx.fill(
        Rect::from_origin_size(Point::ZERO, CHESS_BOARD_SIZE),
        &WHITE_SQUARE_COLOR,
    );
    for i in 0.. 8 {
        for j in 0.. 8 {
            if (i + j) % 2 == 1 { continue };
            ctx.fill(
                sq_rect(i, j),
                &BLACK_SQUARE_COLOR
            );
        }
    }
}

fn draw_piece(svg: &SvgData, ctx: &mut PaintCtx, origin: Point, scale: f64) {
    // svg
    let sq_size = Size::new(SQUARE_SIZE, SQUARE_SIZE);
    let sq = Rect::from_origin_size(origin, sq_size);
    let transform = piet::kurbo::TranslateScale::new(
        sq.origin().to_vec2()
            + Vec2::new(SQUARE_SIZE, SQUARE_SIZE) * ((1.0 - scale) / 2.0),
        (SQUARE_SIZE / 45.0 /* scale to square size */) * scale,
    ).into();
    svg.to_piet(transform, ctx);
}

fn draw_qpiece(game: &ChessGame, ctx: &mut PaintCtx, origin: Point, piece: Piece, qbit: usize) {
    draw_piece(game.svg(piece), ctx, origin, 0.75);

    // Probability circles
    let prob = game.occupancy_prob[qbit] as f64;
    if prob > 0.999 { return };
    let center = origin + (SQUARE_SIZE / 2.0, SQUARE_SIZE / 2.0);
    let angle = prob * TAU;

    let prob_rel =
        game.occupancy_prob_relative.as_ref().map(|m| m[qbit] as f64);
    if prob_rel.map(|p| (p - prob).abs() >= 0.0001).unwrap_or(false) {
        let prob_rel = prob_rel.unwrap();
        let angle_rel = prob_rel * TAU;

        prob_circle(ctx, center, SQUARE_SIZE / 2.0 - 3.0, angle, 127);
        prob_circle(ctx, center, SQUARE_SIZE / 2.0 - 9.0, angle_rel, 255);
    }
    else {
        prob_circle(ctx, center, SQUARE_SIZE / 2.0 - 3.0, angle, 255);
    }
}

fn prob_circle(ctx: &mut PaintCtx, center: Point, radius: f64, angle: f64, alpha: u8) {
    let width_prob = 2.0;
    let width_track = 4.0;

    let track = piet::kurbo::Circle::new(center, radius);
    ctx.stroke(track, &Color::rgba8(30, 30, 30, alpha), width_track);

    let arc = piet::kurbo::CircleSegment {
        center: center,
        outer_radius: radius + width_prob / 2.0,
        inner_radius: radius - width_prob / 2.0,
        start_angle: TAU / 4.0 - angle / 2.0,
        sweep_angle: angle,
    };

    ctx.fill(
        arc,
        &Color::rgba8(255, 40, 255, alpha),
        // 5.0,
    )
}

#[allow(unused_variables)]
impl Widget<ChessGame> for QChessBoard {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, game: &mut ChessGame, env: &Env) {
        use Event::*;

        let get_square = |point: Point| {
            let col_idx = (point.x / SQUARE_SIZE).floor() as usize;
            let mut row_idx = (point.y / SQUARE_SIZE).floor() as usize;
            row_idx = 7 - row_idx;
            let col = Column::from_idx(col_idx);
            let row = Row::from_idx(row_idx);
            col.and_then(|col| row.map(|row| Square{col, row}))
        };

        match event {
            MouseDown(mouse_e) if mouse_e.button.is_left() => {
                let sq = get_square(mouse_e.pos);
                if let Some(sq) = get_square(mouse_e.pos) {
                    if let Some(chess_move) = game.selection.left_click(&*game.state, sq) {
                        if let Ok((new_state, measure)) = game.state.apply_move(chess_move) {
                            // Could use make_mut instead if somtimes this ptr
                            // isn't shared... but I think it always is.
                            new_state.occupancy.print_state();
                            game.state = Rc::new(new_state);
                            game.history.push_back(HistItem{ chess_move, measure });
                            game.occupancy_prob = get_probs(&game.state);
                        }
                        else {
                            println!("Illegal move?")
                        }
                    }
                    game.set_occupancy_prob_rel();
                    ctx.request_paint();
                }
            }
            MouseDown(mouse_e) if mouse_e.button.is_right() => {
                if let Some(sq) = get_square(mouse_e.pos) {
                    game.selection.right_click(sq);
                    game.set_occupancy_prob_rel();
                    ctx.request_paint();
                }
            }
            _ => ()
        };
    }

    fn lifecycle(&mut self, ctx: &mut LifeCycleCtx, event: &LifeCycle, game: &ChessGame, env: &Env) {
    }

    fn update(&mut self, ctx: &mut UpdateCtx, old_game: &ChessGame, game: &ChessGame, env: &Env) {
    }

    fn layout(&mut self, ctx: &mut LayoutCtx, bc: &BoxConstraints, game: &ChessGame, env: &Env) -> Size {
        CHESS_BOARD_SIZE
    }

    fn paint(&mut self, ctx: &mut PaintCtx, game: &ChessGame, env: &Env) {
        draw_board(ctx);

        // Draw pieces
        for i in 0.. 8 {
            for j in 0.. 8 {
                let chess_sq = Square {
                    row: Row::from_idx(i).unwrap(),
                    col: Column::from_idx(j).unwrap(),
                };
                let piece = game.state.pieces[chess_sq.get_idx()];
                if let Some(piece) = piece {
                    draw_qpiece(game, ctx, origin(i, j), piece, i * 8 + j);
                }
            }
        }

        // Draw selection
        let clip_corner_rad = ((SQUARE_SIZE * SQUARE_SIZE) * (1.0/4.0 + 1.0/16.0)).sqrt();
        let sel_color_1 = Color::rgb8(0, 50, 90);
        let sel_color_2 = Color::rgb8(100, 30, 0);
        let sel_width = 4.0;

        let draw_sel = |ctx: &mut PaintCtx, sq: Square, col: &Color| {
            let rect = sq_rect(sq.row.get_idx(), sq.col.get_idx());
            ctx.stroke(rect, col, sel_width);
        };

        let draw_partial_sel = |ctx: &mut PaintCtx, sq: Square, col: &Color| {
            let rect = sq_rect(sq.row.get_idx(), sq.col.get_idx());
            let center = origin(sq.row.get_idx(), sq.col.get_idx()) + (SQUARE_SIZE / 2.0, SQUARE_SIZE / 2.0);
            ctx.with_save(|ctx| {
                // I'd prefer to stroke the *outisde* of the circle, but it's not obvious how to
                // do that with the api so this will suffice for now.
                ctx.clip(Circle::new(center, clip_corner_rad));
                ctx.stroke(rect, col, sel_width);
            })
        };

        match game.selection {
            Selection::None => (),
            Selection::Start(sq) => draw_sel(ctx, sq, &sel_color_1),
            Selection::StartMerge(sq) => draw_partial_sel(ctx, sq, &sel_color_1),
            Selection::MiddleMerge(sq1, sq2) => {
                draw_partial_sel(ctx, sq1, &sel_color_1);
                draw_sel(ctx, sq2, &sel_color_1);
            }
            Selection::StartSplit(from, to) => {
                draw_sel(ctx, from, &sel_color_1);
                draw_partial_sel(ctx, to, &sel_color_2);
            }
        }
    }
}

use qc::apply_monomorphized_fn;
use qcomp::{BasisVector, Complex, QuantumState};

#[derive(Data, Clone)]
struct QChessStateData {
    #[data(ignore)]
    svgs: Rc<[SvgData; 12]>,
    #[data(same_fn = "PartialEq::eq")]
    pieces: [Option<Piece>; 64],
    #[data(same_fn = "PartialEq::eq")]
    occupancy: BasisVector<64>,
    #[data(same_fn = "PartialEq::eq")]
    val: Complex,
}

/// Widget representing a single basis vector
struct QChessStateBoard {}
#[allow(unused_variables)]
impl Widget<QChessStateData> for QChessStateBoard {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, data: &mut QChessStateData, env: &Env) {
    }
    fn lifecycle(&mut self, ctx: &mut LifeCycleCtx, event: &LifeCycle, data: &QChessStateData, env: &Env) {
    }
    fn update(&mut self, ctx: &mut UpdateCtx, old_data: &QChessStateData, data: &QChessStateData, env: &Env) {
    }
    fn layout(&mut self, ctx: &mut LayoutCtx, bc: &BoxConstraints, data: &QChessStateData, env: &Env) -> Size {
        CHESS_BOARD_SIZE * 0.25
    }
    fn paint(
        &mut self,
        ctx: &mut PaintCtx,
        data: &QChessStateData,
        _env: &Env
    ) {
        ctx.transform(Affine::scale(0.25));

        draw_board(ctx);

        for i in 0.. 8 {
            for j in 0.. 8 {
                if !data.occupancy[i * 8 + j] { continue };
                let chess_sq = Square {
                    row: Row::from_idx(i).unwrap(),
                    col: Column::from_idx(j).unwrap(),
                };
                let piece = data.pieces[chess_sq.get_idx()];
                if let Some(piece) = piece {
                    let svg = &data.svgs[svg_idx(piece)];
                    draw_piece(svg, ctx, origin(i, j), 0.9);
                }
            }
        }


        let half_board = SQUARE_SIZE * 4.0;
        let center = Point::new(half_board, half_board);
        let circle = Circle::new(center, half_board);
        ctx.stroke(circle, &Color::rgba8(50, 0, 0, 128), 4.0);

        let x = data.val[0] * half_board + half_board;
        let y = data.val[1] * half_board + half_board;
        let end_point = Point::new(x, y);
        let line = piet::kurbo::Line::new(center, end_point);
        ctx.stroke(line, &Color::rgba8(200, 0, 0, 128), 16.0);
    }
}

impl ChessGame {
    fn captured_pieces<'a>(&'a self, white_pieces: bool) -> impl Iterator<Item = (Piece, usize)> + 'a {
        (64.. self.occupancy_prob.len()).into_iter().filter_map(move |qbit| {
            // We set piece for every capture qubits, also only capture qubits should still
            // matter (but I need to work on the APIs exposing all of this)
            if let Some(piece) = self.state.occupancy.qbits[qbit].piece {
                if piece.is_white != white_pieces { return None };

                let prob = self.occupancy_prob[qbit] as f64;
                if prob == 0.0 || prob > 0.999 { return None };

                Some((piece, qbit))
            }
            else {
                None
            }
        })
    }
}

struct QCaptured {
    white_pieces: bool,
}
impl Widget<ChessGame> for QCaptured {
    fn event(&mut self, _ctx: &mut EventCtx, _event: &Event, _data: &mut ChessGame, _env: &Env) {
        // No input handling for now, maybe some sort of analysis selection
        // eventually, and inspecting when pieces were captured
    }
    fn lifecycle(&mut self, _ctx: &mut LifeCycleCtx, _event: &LifeCycle, _data: &ChessGame, _env: &Env) {
    }
    fn update(&mut self, _ctx: &mut UpdateCtx, _old_data: &ChessGame, _data: &ChessGame, _env: &Env) {
    }
    fn layout(&mut self, _ctx: &mut LayoutCtx, _bc: &BoxConstraints, game: &ChessGame, _env: &Env) -> Size {
        let width = ((game.captured_pieces(self.white_pieces).count() + 7) / 8) as f64 * SQUARE_SIZE;
        let height = 8.0 * SQUARE_SIZE;
        // Apply scale...
        Size::new(width, height) * 0.5
    }
    fn paint(&mut self, ctx: &mut PaintCtx, game: &ChessGame, _: &Env) {
        // TODO: Do I need to restore here?
        // Draw captured pieces at half size
        ctx.transform(Affine::scale(0.5));

        for (idx, (piece, qbit)) in game.captured_pieces(self.white_pieces).enumerate() {
            let mut i = idx % 8;
            if !self.white_pieces { i = 7 - i };
            let j =  idx / 8;
            let origin = Point::new(j as f64 * SQUARE_SIZE, i as f64 * SQUARE_SIZE);
            draw_qpiece(game, ctx, origin, piece, qbit);
        }
    }
}
