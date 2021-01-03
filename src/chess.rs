#![allow(incomplete_features)]
#![feature(
    backtrace,
    const_generics,
    const_evaluatable_checked,
    const_panic,
    array_map,
    hash_drain_filter,
    trait_alias,
    format_args_capture
)]

#[macro_use] pub mod qcomp;
mod qcomp_explainable;
mod gates;

use qcomp::{QuantumState, BasisVector};
use qcomp_explainable::SimpleQuantumState;
use std::fmt;

#[macro_export]
macro_rules! with_const {
    ($name:ident : $ty:ty = $val:expr; $($cval:literal)* $expr:block) => {
        match $val {
            $(
                $cval => {
                    const $name: $ty = $cval;
                    $expr
                }
            ),*
            _ => panic!("Unexpected value in with_const")
        }
    }
}

pub struct ChessState {
    pub occupancy: SimpleQuantumState,
    pub pieces: [Option<Piece>; 64],

    pub is_white_turn: bool,
    // [[white kingside, white queenside], [black kingside, black queenside]]
    pub castling: [[bool; 2]; 2],
    pub en_passant_file: Option<Column>,
    pub move_number: u16,
}

// TODO: Move druid::Data behind a cfg gate or something, also this sucks
#[derive(Copy, Clone, PartialEq, druid::Data)]
pub enum Move {
    Simple{ from: Square, to: Square },
    Split{ from: Square, to_1: Square, to_2: Square },
    Merge{ from_1: Square, from_2: Square, to: Square },
    Promote{ from: Square, to: Square, kind: PieceKind },
}

enum MovementType {
    Jumps, Slides, Pawn
}

#[derive(Debug)]
enum QuantumMove {
    // Optimization potential: Vec is at most 6 long, move fits in a u8,
    // much better to keep this as a inline list...
    Simple{ from: Square, to: Square, path: Vec<Square> },
    Blocked{ from: Square, to: Square, path: Vec<Square> },
    Capture{ from: Square, to: Square, path: Vec<Square> },

    Split{ from: Square, to_1: Square, to_2: Square, path_1: Vec<Square>, path_2: Vec<Square> },
    Merge{ from_1: Square, from_2: Square, to: Square, path_1: Vec<Square>, path_2: Vec<Square> },

    // Pawns are weird
    PawnCapture{ from: Square, to: Square },
    EnPassant{ from: Square, to: Square },
    BlockedEnPassant{from: Square, to: Square},
    CaptureEnPassant{from: Square, to: Square},

    // Castling
    KingCastle,
    QueenCastle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub is_white: bool,
    pub kind: PieceKind,
}
impl Piece {
    pub fn symbol(self) -> char {
        use PieceKind::*;
        match (self.is_white, self.kind) {
            (true, Pawn) => '♙',
            (true, Rook) => '♖',
            (true, Knight) => '♘',
            (true, Bishop) => '♗',
            (true, Queen) => '♕',
            (true, King) => '♔',
            (false, Pawn) => '♟',
            (false, Rook) => '♜',
            (false, Knight) => '♞',
            (false, Bishop) => '♝',
            (false, Queen) => '♛',
            (false, King) => '♚',
        }
    }


    pub fn symbol_black(self) -> char {
        use PieceKind::*;
        match self.kind {
            Pawn => '♟',
            Rook => '♜',
            Knight => '♞',
            Bishop => '♝',
            Queen => '♛',
            King => '♚',
        }
    }
}

// TODO: Move druid::Data behind a cfg gate or something, also this sucks
#[derive(Clone, Copy, PartialEq, Eq, Debug, druid::Data)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    King,
    Queen,
}


// TODO: Move druid::Data behind a cfg gate or something, also this sucks
#[derive(Copy, Clone, PartialEq, Eq, druid::Data)]
pub struct Square {
    #[data(same_fn = "PartialEq::eq")]
    pub col: Column,
    #[data(same_fn = "PartialEq::eq")]
    pub row: Row,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Column {
    A, B, C, D, E, F, G, H
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Row {
    R1, R2, R3, R4, R5, R6, R7, R8
}

impl ChessState {
    pub fn new_game() -> ChessState {
        let mut bv = qcomp::BasisVector::<64>::zero();
        let mut pieces = [None; 64];
        for &color in &[true, false] {
            for &row_perspective in &[Row::R1, Row::R2] {
                // perspective is it's own inverse.
                let row = row_perspective.perspective(color);
                use Column::*;
                use PieceKind::*;
                for &(col, piece_kind) in &[
                    (A, Rook),
                    (B, Knight),
                    (C, Bishop),
                    (D, Queen),
                    (E, King),
                    (F, Bishop),
                    (G, Knight),
                    (H, Rook)
                ] {
                    let square = Square{ row, col };
                    bv.set(square.get_idx(), true);
                    let kind =
                        if row_perspective == Row::R1 { piece_kind }
                        else{ PieceKind::Pawn };
                    pieces[square.get_idx()] = Some(Piece {
                        is_white: color,
                        kind: kind,
                    });
                }
            }
        }

        ChessState {
            occupancy: SimpleQuantumState::new(bv, |i| format!("{}", Square::from_idx(i).unwrap())),
            pieces,
            is_white_turn: true,
            castling: [[true, true], [true, true]],
            en_passant_file: None,
            move_number: 0,
        }
    }

    /// Returns Result<Measurement State, Error String>
    /// and mutates the state on Ok.
    pub fn apply_move(&self, chess_move: Move) -> Result<
        (ChessState, Option<bool>),
        String
    > {
        fn remove_zero_prob_pieces<const N: usize>(
            state: &QuantumState<N>,
            pieces: &mut [Option<Piece>; 64]
        )
        where [u8; (N + 7)/8]: {
            let mut used = BasisVector::<64>::zero();
            for (bv, _) in state.iter() {
                used |= bv.resize::<64>();
            }
            for i in 0.. 64 {
                if !used[i] {
                    pieces[i] = None;
                }
            }
        }

        if self.is_legal(chess_move) {
            let qmove = self.get_qmove(chess_move);
            println!("\nMove is a {:?}", qmove);
            let (occupancy, mut pieces, measurement) = self.compute_move(qmove);
            if &self.occupancy.state == &occupancy.state {
                Err("State didn't change, try again".into())
            }
            else {
                // En passant
                let en_passant_file = if let Move::Simple{from, to} = chess_move {
                    if from.row.perspective(self.is_white_turn) == Row::R2
                    && to.row.perspective(self.is_white_turn) == Row::R4
                    && self.pieces[from.get_idx()].map(|piece| piece.kind) == Some(PieceKind::Pawn) {
                        Some(from.col)
                    } else { None }
                } else { None };

                // Promotion
                if let Move::Promote{to, kind, ..} = chess_move {
                    if pieces[to.get_idx()] == Some(Piece{ kind: PieceKind::Pawn, is_white: self.is_white_turn }) {
                        pieces[to.get_idx()] = Some(Piece{ kind, is_white: self.is_white_turn });
                    }
                }

                let mut new_state = ChessState {
                    occupancy,
                    pieces,

                    is_white_turn: !self.is_white_turn,
                    castling: chess_move.update_castling(self.is_white_turn, self.castling),
                    // TODO:
                    en_passant_file,
                    move_number: self.move_number + 1,
                };
                apply_monomorphized_fn!(&new_state.occupancy.state, remove_zero_prob_pieces, &mut new_state.pieces);
                Ok((new_state, measurement))
            }
        }
        else {
            Err("Illegal move".into())
        }
    }

    fn get_qmove(&self, chess_move: Move) -> QuantumMove {
        use QuantumMove::*;
        use MovementType::{Slides, Jumps, Pawn as PawnMove};
        let kind = self.get_moves_piece_kind(chess_move);

        // Castling
        if kind == PieceKind::King {
            if let Move::Simple{ from, to } = chess_move {
                if from.col.dist(to.col) == 2 {
                    if to.col == Column::G {
                        return KingCastle
                    }
                    else {
                        return QueenCastle
                    }
                }
            }
        }

        // Every other type of move
        match (kind.movement_type(), chess_move) {
            // Simple moves
            (Jumps, Move::Simple{ from, to }) => {
                if self.pieces[from.get_idx()] == self.pieces[to.get_idx()] {
                    return Simple{ from, to, path: vec![] }
                }

                match self.get_relative_team(to) {
                    Some(true) => Blocked{ from, to, path: vec![] },
                    Some(false) => Capture{ from, to, path: vec![] },
                    None => Simple{ from, to, path: vec![] },
                }
            },
            (Slides, Move::Simple{ from, to }) => {
                let path = from.path(to, None);
                if self.pieces[from.get_idx()] == self.pieces[to.get_idx()] {
                    return Simple{ from, to, path }
                }

                match self.get_relative_team(to) {
                    Some(true) => Blocked{ from, to, path },
                    Some(false) => Capture{ from, to, path },
                    None => Simple{ from, to, path },
                }
            },
            (PawnMove, Move::Simple{ from, to }) => {
                let path = from.path(to, None);
                if self.pieces[from.get_idx()] == self.pieces[to.get_idx()] {
                    return Simple{ from, to, path }
                }

                let is_en_passant =
                    from.row.perspective(self.is_white_turn) == Row::R5
                    && from.col.dist(to.col) == 1
                    && Some(to.col) == self.en_passant_file;
                let is_capture = from.col != to.col;
                let team = self.get_relative_team(to);
                match (is_capture, is_en_passant, team) {
                    (false, false, None) => Simple{ from, to, path },
                    (false, false, Some(_)) => Blocked{ from, to, path },
                    (true, false, Some(false)) => PawnCapture{ from, to },
                    (true, true, None) => EnPassant{ from, to },
                    (true, true, Some(true)) => BlockedEnPassant{ from, to },
                    (true, true, Some(false)) => CaptureEnPassant{ from, to },
                    // Error checking for completeness
                    (false, true, _) => panic!("en_passsant must capture"),
                    (true, false, _) => panic!("Must be capturing a piece")
                }
            }

            // Split moves
            (Jumps, Move::Split{ from, to_1, to_2 }) =>
                Split { from, to_1, to_2, path_1: vec![], path_2: vec![] },
            (Slides, Move::Split{ from, to_1, to_2 }) => {
                let path_1 = from.path(to_1, Some(to_2));
                let path_2 = from.path(to_2, Some(to_1));
                Split{ from, to_1, to_2, path_1, path_2 }
            },

            // Merge moves
            (Jumps, Move::Merge{ from_1, from_2, to }) =>
                Merge{ from_1, from_2, to, path_1: vec![], path_2: vec![] },
            (Slides, Move::Merge{ from_1, from_2, to }) => {
                let path_1 = from_1.path(to, Some(from_2));
                let path_2 = from_2.path(to, Some(from_1));
                Merge{ from_1, from_2, to, path_1, path_2 }
            },

            // Promotion
            // Note: Promotion happens as a standard pawn quantum move,
            // and then a seperate promotion after the fact.
            (PawnMove, Move::Promote{ from, to, .. }) => {
                match (from.col == to.col, self.get_type(to).is_none()) {
                    (true, true) => Simple{ from, to, path: vec![] },
                    (true, false) => Blocked{ from, to, path: vec![] },
                    (false, true) => unreachable!(),
                    (false, false) => Capture{ from, to, path: vec![] },
                }
            },

            // Trying to merge/split a pawn, or promote a non pawn.
            // These are caught by is_illegal so shouldn't happen here.
            _ => unreachable!()
        }
    }

    // Note: *Very* large function despite it's apparently only reasonably large size,
    // we're using macros here to expand the line count by roughly 49 times!
    // If codegen has an issue, that's *probably* why.
    fn compute_move(&self, qmove: QuantumMove) -> (SimpleQuantumState, [Option<Piece>; 64], Option<bool>) {
        use QuantumMove::*;
        match qmove {
            Simple{ from, to, path } => {
                let gate = gates::iswap();
                let indicies = [from.get_idx(), to.get_idx()];
                let occupancy = with_const!(LEN: usize = path.len(); 0 1 2 3 4 5 6 {
                    let mut path_idx = [0; LEN];
                    for i in 0.. LEN {
                        path_idx[i] = path[i].get_idx();
                    }
                    self.occupancy.apply_controlled_unitaries_and_clone(
                        [(&gate, None, &[], &path_idx)],
                        indicies,
                    )
                });

                let mut pieces = self.pieces;
                pieces[to.get_idx()] = pieces[from.get_idx()];

                (occupancy, pieces, None)
            }
            Blocked{ from, to, path } => {
                // Temporary ancilla for measurement, false only if we failed to move
                // *because* we were blocked by a piece in target.
                let (mut occupancy, not_blocked) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("Block_{from}{to}"),
                    format!("Ancilla qbit to measure whether {to} is occupied blocking {from}{to}")
                );
                occupancy
                    .apply_controlled_unitaries(
                        [(&gates::pauli_x(), None, &[], &[to.get_idx()])],
                        [not_blocked],
                    );
                let bit_not_blocked = occupancy.measure(not_blocked);
                occupancy.free_qubit(not_blocked);
                let mut pieces = self.pieces;
                if bit_not_blocked {
                    with_const!(LEN: usize = path.len(); 0 1 2 3 4 5 6 {
                        let mut path_idx = [0; LEN];
                        for i in 0.. LEN {
                            path_idx[i] = path[i].get_idx();
                        }
                        occupancy.apply_controlled_unitaries(
                            [(&gates::iswap(), None, &[], &path_idx)],
                            [from.get_idx(), to.get_idx()],
                        )
                    });
                    pieces[to.get_idx()] = pieces[from.get_idx()];
                }

                (occupancy, pieces, Some(bit_not_blocked))
            }
            Capture{ from, to, path } => {
                // Temporary ancilla for measurement, true if either we sucesfully moved
                // or the target is unoccupied.
                let purpose =
                    if path.len() == 0 {
                        format!("Ancilla qbit to measure whether {from} is occupied so {from}{to} captures a piece")
                    } else {
                        let path_descr: String = path.iter().map(|v| format!("{} ", v)).collect::<String>();
                        let path_descr = &path_descr[0.. path_descr.len() - 1];
                        format!("Ancilla qibt to measure whether either:\n 1. {from} is occupied and the path ({path_descr}) is not so the capture {from}{to} succeeds, or\n 2. The path ({path_descr}) is blocked but {to} is unocuppied so the capture didn't need to suceed")
                    };
                let (mut occupancy, still_there) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("CanCapture_{from}{to}"),
                    purpose
                );

                let bit = with_const!(LEN: usize = path.len(); 0 1 2 3 4 5 6 {
                    let mut path_indicies  = [0; LEN];
                    for i in 0.. LEN {
                        path_indicies[i] = path[i].get_idx()
                    }

                    // TODO: Merge these calls for performance
                    occupancy
                        .apply_controlled_unitaries(
                            [(&gates::pauli_x(), Some(&[from.get_idx()]), &[], &path_indicies)],
                            [still_there],
                        );
                    occupancy
                        .apply_controlled_unitaries(
                            [(&gates::pauli_x(), Some(&path_indicies), &[], &[to.get_idx()])],
                            [still_there],
                        );

                    let bit = occupancy.measure(still_there);
                    occupancy.free_qubit(still_there);

                    if bit {
                        // Qubit that might stick arround for awhile, to represent whether or
                        // not the piece was captured
                        let sym = self.pieces[to.get_idx()].unwrap().symbol();
                        let is_captured = occupancy.alloc_qubit(
                            self.pieces[to.get_idx()],
                            format!("Captured{sym}_{from}{to}"),
                            format!("Whether the {sym} formerly on {to} was captured")
                        );
                        occupancy.apply_controlled_unitaries(
                            [(&gates::iswap3_01_12(), None, &[], &path_indicies)],
                            [is_captured, to.get_idx(), from.get_idx()]
                        );
                        occupancy.free_qubit(is_captured);
                    }

                    bit
                });

                let mut pieces = self.pieces;
                if bit {
                    pieces[to.get_idx()] = pieces[from.get_idx()];
                }

                (occupancy, pieces, Some(bit))
            }

            Split{ from, to_1, to_2, path_1, path_2 } => {
                let indicies_split = [from.get_idx(), to_1.get_idx(), to_2.get_idx()];

                let gate_split = gates::qc_split();
                let gate_jmp_1 = gates::iswap3_01();
                let gate_jmp_2 = gates::iswap3_02();

                let occupancy = with_const!(LEN_2: usize = path_2.len(); 0 1 2 3 4 5 6 {
                    let mut path_2_idx = [0; LEN_2];
                    for i in 0.. LEN_2 {
                        path_2_idx[i] = path_2[i].get_idx();
                    }
                    with_const!(LEN_1: usize = path_1.len(); 0 1 2 3 4 5 6 {
                        let mut path_1_idx = [0; LEN_1];
                        let mut path_split_idx = [0; LEN_1 + LEN_2];
                        for i in 0.. LEN_1 {
                            path_1_idx[i] = path_1[i].get_idx();
                            path_split_idx[i] = path_1[i].get_idx();
                        }
                        for i in 0.. LEN_2 {
                            path_split_idx[LEN_1 + i] = path_2_idx[i];
                        }

                        self.occupancy.apply_controlled_unitaries_and_clone(
                            [
                                (
                                    &gate_split,
                                    None,
                                    &[],
                                    &path_split_idx as &[usize]
                                ),
                                (
                                    &gate_jmp_1,
                                    Some(&path_2_idx as &[usize]),
                                    &[],
                                    &path_1_idx
                                ),
                                (
                                    &gate_jmp_2,
                                    Some(&path_1_idx),
                                    &[],
                                    &path_2_idx
                                )
                            ],
                            indicies_split
                        )
                    })
                });

                let mut pieces = self.pieces;
                pieces[to_1.get_idx()] = pieces[from.get_idx()];
                pieces[to_2.get_idx()] = pieces[from.get_idx()];

                (occupancy, pieces, None)
            }
            Merge{ from_1, from_2, to, path_1, path_2 } => {
                // let indicies_merge = [from_1.get_idx(), from_2.get_idx(), to.get_idx()];
                let indicies_merge = [to.get_idx(), from_2.get_idx(), from_1.get_idx()];

                let gate_merge = gates::qc_merge();
                let gate_jmp_2 = gates::iswap3_01();
                let gate_jmp_1 = gates::iswap3_02();

                let occupancy = with_const!(LEN_2: usize = path_2.len(); 0 1 2 3 4 5 6 {
                    let mut path_2_idx = [0; LEN_2];
                    for i in 0.. LEN_2 {
                        path_2_idx[i] = path_2[i].get_idx();
                    }
                    with_const!(LEN_1: usize = path_1.len(); 0 1 2 3 4 5 6 {
                        let mut path_1_idx = [0; LEN_1];
                        let mut path_merge_idx = [0; LEN_1 + LEN_2];
                        for i in 0.. LEN_1 {
                            path_1_idx[i] = path_1[i].get_idx();
                            path_merge_idx[i] = path_1[i].get_idx();
                        }
                        for i in 0.. LEN_2 {
                            path_merge_idx[LEN_1 + i] = path_2_idx[i];
                        }

                        self.occupancy.apply_controlled_unitaries_and_clone(
                            [
                                (
                                    &gate_merge,
                                    None,
                                    &[],
                                    &path_merge_idx,
                                ),
                                (
                                    &gate_jmp_1,
                                    Some(&path_2_idx),
                                    &[],
                                    &path_1_idx,
                                ),
                                (
                                    &gate_jmp_2,
                                    Some(&path_1_idx),
                                    &[],
                                    &path_2_idx,
                                )

                            ],
                            indicies_merge
                        )
                    })
                });

                let mut pieces = self.pieces;
                pieces[to.get_idx()] = pieces[from_1.get_idx()];

                (occupancy, pieces, None)
            }

            PawnCapture{ from, to } => {
                let (mut occupancy, pawn_present) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("CanPawnCapture_{from}{to}"),
                    format!("Ancilla qbit to measure whether {from} is actually there")
                );
                occupancy.apply_controlled_unitaries(
                    [(&gates::pauli_x(), Some(&[from.get_idx()]), &[], &[])],
                    [pawn_present],
                );
                let bit_present = occupancy.measure(pawn_present);
                occupancy.free_qubit(pawn_present);

                if bit_present {
                    let sym = self.pieces[to.get_idx()].unwrap().symbol();
                    let captured = occupancy.alloc_qubit(
                        self.pieces[to.get_idx()],
                        format!("PawnCaptured{sym}_{from}{to}"),
                        format!("Whether the {sym} formerly on {to} was captured")
                    );
                    occupancy.apply_controlled_unitaries(
                        // Mild deviation from the paper, which asks for both from and to,
                        // but we only allow for "any" requirements here not all, and we
                        // just measured from to be there, so this is equivalent.
                        [(&gates::iswap3_01_12(), Some(&[to.get_idx()]), &[], &[])],
                        [captured, to.get_idx(), from.get_idx()],
                    );
                    occupancy.free_qubit(captured);
                    let mut pieces = self.pieces;
                    pieces[to.get_idx()] = pieces[from.get_idx()];
                    (occupancy, pieces, Some(true))
                }
                else {
                    (occupancy, self.pieces, Some(false))
                }
            }

            KingCastle => {
                use Column::*;
                let row = Row::R1.perspective(self.is_white_turn);
                // king from, king to, rook from, rook to
                let sq_kf = Square{ row, col: E };
                let sq_kt = Square{ row, col: G };
                let sq_rf = Square{ row, col: H };
                let sq_rt = Square{ row, col: F };

                let (mut occupancy, can_castle) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("CanKingCastle"),
                    format!("Ancilla qubit to measure whether or not castling suceeds")
                );
                occupancy.apply_controlled_unitaries(
                    [(&gates::pauli_x(), None, &[], &[sq_kt.get_idx(), sq_rt.get_idx()])],
                    [can_castle],
                );
                let can_castle_bit = occupancy.measure(can_castle);
                occupancy.free_qubit(can_castle);
                let mut pieces = self.pieces;
                if can_castle_bit {
                    occupancy.apply_unitary(
                        gates::iswap(),
                        [sq_rf.get_idx(), sq_rt.get_idx()],
                    );
                    occupancy.apply_unitary(
                        gates::iswap(),
                        [sq_kf.get_idx(), sq_kt.get_idx()],
                    );
                    pieces[sq_rt.get_idx()] = pieces[sq_rf.get_idx()];
                    pieces[sq_kt.get_idx()] = pieces[sq_kf.get_idx()];
                }

                (occupancy, pieces, Some(can_castle_bit))
            }

            QueenCastle => {
                use Column::*;
                let row = Row::R1.perspective(self.is_white_turn);
                // king from, king to, rook from, rook to
                let sq_kf = Square{ row, col: E };
                let sq_kt = Square{ row, col: C };
                let sq_rf = Square{ row, col: A };
                let sq_rt = Square{ row, col: D };
                let sq_mid = Square{ row, col: B };

                let (mut occupancy, can_castle) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("CanQueenCastle"),
                    format!("Ancilla qubit to measure whether or not the squares we move to while castling are available")
                );
                occupancy.apply_controlled_unitaries(
                    [(&gates::pauli_x(), None, &[], &[sq_kt.get_idx(), sq_rt.get_idx()])],
                    [can_castle],
                );
                let can_castle_bit = occupancy.measure(can_castle);
                occupancy.free_qubit(can_castle);
                let mut pieces = self.pieces;
                if can_castle_bit {
                    occupancy.apply_controlled_unitaries(
                        [(&gates::iswap(), None, &[], &[sq_mid.get_idx()])],
                        [sq_rf.get_idx(), sq_rt.get_idx()],
                    );
                    occupancy.apply_controlled_unitaries(
                        [(&gates::iswap(), None, &[], &[sq_mid.get_idx()])],
                        [sq_kf.get_idx(), sq_kt.get_idx()],
                    );
                    pieces[sq_rt.get_idx()] = pieces[sq_rf.get_idx()];
                    pieces[sq_kt.get_idx()] = pieces[sq_kf.get_idx()];
                }

                (occupancy, pieces, Some(can_castle_bit))
            }

            EnPassant{ from, to } => {
                let ep = Square{
                    col: to.col,
                    row: Row::R5.perspective(self.is_white_turn),
                };

                let sym = self.pieces[ep.get_idx()].unwrap().symbol();
                let (mut occupancy, capture) = self.occupancy.alloc_qubit_and_clone(
                    self.pieces[ep.get_idx()],
                    format!("Captured{sym}_{from}{to}EnPassant"),
                    format!("Whether the {sym} formerly on {ep} was captured via en passant")
                );

                occupancy.apply_controlled_unitaries(
                    [(&gates::iswap4_01_23(), None, &[from.get_idx(), ep.get_idx()], &[])],
                    [from.get_idx(), to.get_idx(), ep.get_idx(), capture],
                );

                let mut pieces = self.pieces;
                pieces[to.get_idx()] = pieces[from.get_idx()];
                (occupancy, pieces, None)
            }

            BlockedEnPassant{ from, to } => {
                let ep = Square{
                    col: to.col,
                    row: Row::R5.perspective(self.is_white_turn),
                };

                let (mut occupancy, ancilla) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("EnPassantNotBlocked"),
                    format!("Whether the piece on {to} does not block en passant")
                );
                occupancy.apply_controlled_unitaries(
                    [(&gates::pauli_x(), None, &[], &[to.get_idx()])],
                    [ancilla]
                );
                let can_enpassant = occupancy.measure(ancilla);

                let mut pieces = self.pieces;
                if can_enpassant {
                    let sym = self.pieces[ep.get_idx()].unwrap().symbol();
                    let capture = occupancy.alloc_qubit(
                        self.pieces[ep.get_idx()],
                        format!("Captured{sym}_{from}{to}EnPassant"),
                        format!("Whether the {sym} formerly on {ep} was captured via en passant")
                    );

                    occupancy.apply_controlled_unitaries(
                        [(&gates::iswap4_01_23(), None, &[from.get_idx(), ep.get_idx()], &[])],
                        [from.get_idx(), to.get_idx(), ep.get_idx(), capture],
                    );

                    pieces[to.get_idx()] = pieces[from.get_idx()];
                }

                (occupancy, pieces, Some(can_enpassant))
            }

            CaptureEnPassant{ from, to } => {
                let ep = Square{
                    col: to.col,
                    row: Row::R5.perspective(self.is_white_turn),
                };

                let (mut occupancy, ancilla) = self.occupancy.alloc_qubit_and_clone(
                    None,
                    format!("EnPassantFromThere"),
                    format!("Whether the piece on {from} is actually there to execute the capture/en passant")
                );
                occupancy.apply_controlled_unitaries(
                    [(&gates::pauli_x(), None, &[from.get_idx()], &[])],
                    [ancilla]
                );
                let can_enpassant = occupancy.measure(ancilla);

                let mut pieces = self.pieces;
                if can_enpassant {
                    let sym = self.pieces[ep.get_idx()].unwrap().symbol();
                    let capture_ep = occupancy.alloc_qubit(
                        self.pieces[ep.get_idx()],
                        format!("Captured{sym}_{from}{to}EnPassant"),
                        format!("Whether the {sym} formerly on {ep} was captured via en passant")
                    );

                    let sym = self.pieces[to.get_idx()].unwrap().symbol();
                    let capture = occupancy.alloc_qubit(
                        self.pieces[to.get_idx()],
                        format!("Captured{sym}_{from}{to}"),
                        format!("Whether the {sym} formerly on {ep} was captured (instead of en passant)")
                    );

                    occupancy.apply_controlled_unitaries(
                        [(&gates::iswap5_43_21_10(), Some(&[to.get_idx(), ep.get_idx()]), &[], &[])],
                        [from.get_idx(), to.get_idx(), capture, ep.get_idx(), capture_ep],
                    );

                    pieces[to.get_idx()] = pieces[from.get_idx()];
                }

                (occupancy, pieces, Some(can_enpassant))
            }
        }
    }

    /// Checks that the pieces only move in "piece appropriate ways".
    ///
    /// Including:
    ///
    /// 1. Pawns only take pieces of the opposite color.
    /// 2. Pieces only move to squares they could move to in traditional chess on an empty board.
    /// 3. Castling rules.
    /// 4. It is the right turn for the moving piece to move.
    ///
    /// This *does not* include any checks on whether or not the piece moves
    /// through any other pieces, that happens later at a measurement phase.
    fn is_legal(&self, chess_move: Move) -> bool {
        match chess_move {
            Move::Simple{from, to } => {
                if from == to {
                    return false;
                }

                let piece_kind = match self.get_type(from) {
                    Some(Piece{kind, is_white}) if is_white == self.is_white_turn => { kind }
                    _ => return false
                };

                match piece_kind {
                    PieceKind::Pawn => {
                        // 4 legal types of moves.
                        // 1. Moving forward 1
                        // 2. Moving forward 2 if on second row
                        // 3. Moving diagonal one if that square is of the opposing team
                        // 4. en passant

                        // Type 1
                        if from.col == to.col
                        && from.row.forwards(1, self.is_white_turn) == Some(to.row) {
                            return true
                        }

                        // Type 2
                        if from.col == to.col
                        && from.row.perspective(self.is_white_turn) == Row::R2
                        && from.row.forwards(2, self.is_white_turn) == Some(to.row) {
                            return true
                        }

                        // Type 3
                        if from.col.dist(to.col) == 1
                        && from.row.forwards(1, self.is_white_turn) == Some(to.row)
                        && self.get_team(to) == Some(!self.is_white_turn) {
                            return true
                        }

                        // Type 4
                        if dbg!(from.col.dist(to.col) == 1)
                        && dbg!(from.row.perspective(self.is_white_turn) == Row::R5)
                        && dbg!(to.row.perspective(self.is_white_turn) == Row::R6)
                        && dbg!(Some(to.col) == self.en_passant_file) {
                            println!("En passant");
                            return true
                        }

                        false
                    }
                    PieceKind::King => {
                        // King Side Castling
                        if from.col == Column::E && to.col == Column::G
                        && self.can_king_side_castle()
                        && from.row.perspective(self.is_white_turn) == Row::R1
                        && to.row.perspective(self.is_white_turn) == Row::R1 {
                            return true;
                        }

                        // Queen side castling
                        if from.col == Column::E && to.col == Column::C
                        && self.can_queen_side_castle()
                        && from.row.perspective(self.is_white_turn) == Row::R1
                        && to.row.perspective(self.is_white_turn) == Row::R1 {
                            return true;
                        }

                        from.col.dist(to.col) <= 1 && from.row.dist(to.row) <= 1
                    }
                    PieceKind::Knight => {
                        (from.col.dist(to.col) == 2 && from.row.dist(to.row) == 1) ||
                            (from.row.dist(to.row) == 2 && from.col.dist(to.col) == 1)
                    }
                    PieceKind::Bishop => {
                        from.col.dist(to.col) == from.row.dist(to.row)
                    }
                    PieceKind::Rook => {
                        (from.col.dist(to.col) == 0 && from.row.dist(to.row) != 0) ||
                            (from.row.dist(to.row) == 0 && from.col.dist(to.col) != 0 )
                    }
                    PieceKind::Queen => {
                        (from.col.dist(to.col) == from.row.dist(to.row)) ||
                            (from.col.dist(to.col) == 0 && from.row.dist(to.row) != 0) ||
                            (from.row.dist(to.row) == 0 && from.col.dist(to.col) != 0 )
                    }
                }
            }
            Move::Promote{from, to, kind} => {
                if from == to {
                    return false;
                }
                // Check we are moving to the last row.
                if to.row.perspective(self.is_white_turn) != Row::R8 {
                    return false;
                }
                // Check we are moving a pawn
                match self.get_type(from) {
                    Some(Piece{ kind: PieceKind::Pawn, .. }) => (),
                    _ => return false,
                };
                // Check that we aren't promoting to a pawn
                if kind == PieceKind::Pawn {
                    return false;
                }
                // Check it's an otherwise legal move
                self.is_legal(Move::Simple{from, to})
            }
            Move::Split{ from, to_1, to_2 } => {
                if from == to_1 || from == to_2 || to_1 == to_2 {
                    return false;
                }
                let from_type = match self.get_type(from) {
                    Some(t) => t,
                    None => return false,
                };

                from_type.kind != PieceKind::Pawn
                && self.get_type(to_1).unwrap_or(from_type) == from_type
                && self.get_type(to_2).unwrap_or(from_type) == from_type
                && self.is_legal(Move::Simple{from, to: to_1})
                && self.is_legal(Move::Simple{from, to: to_2})
            }
            Move::Merge{ from_1, from_2, to } => {
                if from_1 == from_2 || from_1 == to || from_2 == to {
                    return false;
                }
                let from_type = match self.get_type(from_1) {
                    Some(t) => t,
                    None => return false,
                };

                from_type.kind != PieceKind::Pawn
                && self.get_type(from_2) == Some(from_type)
                && self.get_type(to).unwrap_or(from_type) == from_type
                && self.is_legal(Move::Simple{from: from_1, to})
                && self.is_legal(Move::Simple{from: from_2, to})
            }
        }
    }

    fn can_king_side_castle(&self) -> bool {
        self.castling[if self.is_white_turn { 0 } else { 1 }][0]
    }

    fn can_queen_side_castle(&self) -> bool {
        self.castling[if self.is_white_turn { 0 } else { 1 }][1]
    }

    // fn get_partition(&self, chess_move: Move) -> impl Fn(BasisVector<64>) -> bool {
    //     let from = match chess_move {
    //         Move::Simple{ from, .. } => Some(from),
    //         Move::Promote{ from, .. } => Some(from),
    //         _ => None
    //     };
    //     move |bv| {
    //         let from = match from {
    //             // Quantum move, no partitioning/always succeeds
    //             None => return true,
    //             Some(from) => from
    //         };

    //         let idx = from.get_idx();
    //         bv[from]
    //     }
    // }

    fn get_type(&self, square: Square) -> Option<Piece> {
        self.pieces[square.get_idx()]
    }

    fn get_moves_piece_kind(&self, chess_move: Move) -> PieceKind {
        use Move::*;
        let from = match chess_move {
            Simple{ from, .. }
            | Promote{ from, .. }
            | Split{ from, .. }
            | Merge { from_1: from, .. } => from
        };
        self.pieces[from.get_idx()].map(|x| x.kind).expect("Get move piece kind only valid if move is legal")
    }

    fn get_team(&self, square: Square) -> Option<bool> {
        self.get_type(square).map(|x| x.is_white)
    }

    fn get_relative_team(&self, square: Square) -> Option<bool> {
        self.get_team(square).map(|x| x == self.is_white_turn)
    }
}

impl PieceKind {
    fn movement_type(self) -> MovementType {
        use PieceKind::*;
        match self {
            Bishop | Rook | Queen => MovementType::Slides,
            Knight | King => MovementType::Jumps,
            Pawn => MovementType::Pawn
        }
    }
}

impl Square {
    pub fn get_idx(self) -> usize {
        let c = self.col.get_idx();
        let r = self.row.get_idx();
        8 * r + c
    }

    fn from_idx(idx: usize) -> Option<Self> {
        let col = Column::from_idx(idx % 8).unwrap();
        Row::from_idx(idx / 8).map(|row| Square{ row, col })
    }

    fn path(self, to: Square, exclude: Option<Square>) -> Vec<Square> {
        let mut cur_row = self.row.get_idx();
        let mut cur_col = self.col.get_idx();
        let mut out = vec![];
        let to_row = to.row.get_idx();
        let to_col = to.col.get_idx();
        loop {
            if cur_row < to_row { cur_row += 1 }
            if cur_row > to_row { cur_row -= 1 }
            if cur_col < to_col { cur_col += 1 }
            if cur_col > to_col { cur_col -= 1 }
            if cur_row == to_row && cur_col == to_col { return out };
            let sq = Square{
                row: Row::from_idx(cur_row).unwrap(),
                col: Column::from_idx(cur_col).unwrap(),
            };
            if exclude == Some(sq) { continue };
            out.push(sq)
        }
    }
}

impl Column {
    fn dist(self, rhs: Column) -> usize {
        (self.get_idx() as i32 - rhs.get_idx() as i32).abs() as usize
    }

    pub fn get_idx(self) -> usize {
        use Column::*;
        match self {
            A => 0,
            B => 1,
            C => 2,
            D => 3,
            E => 4,
            F => 5,
            G => 6,
            H => 7,
        }
    }

    pub fn from_idx(idx: usize) -> Option<Self> {
        use Column::*;
        match idx {
            0 => Some(A),
            1 => Some(B),
            2 => Some(C),
            3 => Some(D),
            4 => Some(E),
            5 => Some(F),
            6 => Some(G),
            7 => Some(H),
            _ => None
        }
    }
}

impl Row {
    fn dist(self, rhs: Row) -> usize {
        (self.get_idx() as i32 - rhs.get_idx() as i32).abs() as usize
    }

    pub fn get_idx(self) -> usize {
        use Row::*;
        match self {
            R1 => 0,
            R2 => 1,
            R3 => 2,
            R4 => 3,
            R5 => 4,
            R6 => 5,
            R7 => 6,
            R8 => 7,
        }
    }

    pub fn from_idx(idx: usize) -> Option<Self> {
        use Row::*;
        match idx {
            0 => Some(R1),
            1 => Some(R2),
            2 => Some(R3),
            3 => Some(R4),
            4 => Some(R5),
            5 => Some(R6),
            6 => Some(R7),
            7 => Some(R8),
            _ => None
        }
    }

    fn forwards(self, n: usize, is_white: bool) -> Option<Row>{
        let idx = self.get_idx();
        if idx < n && !is_white { None }
        else {
            let new_idx = if is_white { n + idx } else { idx - n};
            Self::from_idx(new_idx)
        }
    }

    pub fn perspective(self, is_white: bool) -> Row {
        if is_white{ self }
        else { Self::from_idx(7 - self.get_idx()).unwrap() }
    }
}

impl Square {
    fn parse(str: &mut &str) -> Result<Square, String> {
        use Column::*;
        use Row::*;
        if str.len() < 2 {
            return Err(format!("String '{}' is too short to specify a square", *str));
        }
        let col = match str.as_bytes()[0] {
            b'a' => A,
            b'b' => B,
            b'c' => C,
            b'd' => D,
            b'e' => E,
            b'f' => F,
            b'g' => G,
            b'h' => H,
            _ => return Err(format!("Invalid column: {}", str.chars().next().unwrap()))
        };
        let row = match str.as_bytes()[1] {
            b'1' => R1,
            b'2' => R2,
            b'3' => R3,
            b'4' => R4,
            b'5' => R5,
            b'6' => R6,
            b'7' => R7,
            b'8' => R8,
            _ => return Err(format!("Invalid column: {}", str.chars().next().unwrap()))
        };
        *str = &str[2..];
        Ok(Square{ row, col })
    }
}

impl Column {
    fn get_char(self) -> char {
        use Column::*;
        match self {
            A => 'a',
            B => 'b',
            C => 'c',
            D => 'd',
            E => 'e',
            F => 'f',
            G => 'g',
            H => 'h',
        }
    }
}

impl Row {
    fn get_char(self) -> char {
        use Row::*;
        match self {
            R1 => '1',
            R2 => '2',
            R3 => '3',
            R4 => '4',
            R5 => '5',
            R6 => '6',
            R7 => '7',
            R8 => '8',
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buf = [0; 4];
        fmt.write_str(self.symbol().encode_utf8(&mut buf))
    }
}
impl fmt::Debug for Piece {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buf = [0; 4];
        fmt.write_str(self.symbol().encode_utf8(&mut buf))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_fmt(format_args!("{}{}", self.col.get_char(), self.row.get_char()))
    }
}
impl fmt::Debug for Square {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_fmt(format_args!("{}{}", self.col.get_char(), self.row.get_char()))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use Move::*;
        match *self {
            Simple{ from, to } => {
                fmt.write_fmt(format_args!("{from}{to}"))
            },
            Split{ from, to_1, to_2 } => {
                fmt.write_fmt(format_args!("{from}^{to_1}{to_2}"))
            },
            Merge{ from_1, from_2, to } => {
                fmt.write_fmt(format_args!("{from_1}{from_2}^{to}"))
            }
            Promote{ from, to, kind } => {
                let sym = Piece{ kind, is_white: true }.symbol();
                fmt.write_fmt(format_args!("{from}{to}{sym}"))
            }
        }
    }
}
impl fmt::Debug for Move {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

pub fn get_probs(state: &ChessState) -> Vec<f32> {
    // fn get_prob_matrix_monomorphized<const N: usize>(state: &QuantumState<N>) -> aljabar::Matrix<f32, 8, 8>
    // where [u8; (N + 7) / 8]: {
    //     use aljabar::Zero;
    //     let mut out = aljabar::Matrix::zero();
    //     for (bv, val) in state.iter() {
    //         let prob = val.norm_squared();
    //         for i in 0.. 64 {
    //             if bv[i] {
    //                 let sq = Square::from_idx(i).unwrap();
    //                 out[sq.col.get_idx()][sq.row.get_idx()] += prob as f32;
    //             }
    //         }
    //     }
    //     out
    // }

    // apply_monomorphized_fn!(&state.occupancy.state, get_prob_matrix_monomorphized, )
    get_probs_given(state, &[])
}

pub fn get_probs_given(state: &ChessState, all_true: &[usize]) -> Vec<f32> {
    fn get_prob_matrix_monomorphized<const N: usize>(state: &QuantumState<N>, all_true: &[usize]) -> Vec<f32>
    where [u8; (N + 7) / 8]: {
        let all_true_bv = all_true.into();
        let mut out = vec![0.0; N];
        for (bv, val) in state.iter().filter(|&(bv, _)| (bv & all_true_bv) == all_true_bv) {
            let prob = val.norm_squared();
            for i in 0.. N {
                if bv[i] {
                    out[i] += prob as f32;
                }
            }
        }
        out
    }

    apply_monomorphized_fn!(&state.occupancy.state, get_prob_matrix_monomorphized, all_true)
}

impl Move {
    pub fn parse(str: &mut &str) -> Result<Move, String> {
        let sq1 = Square::parse(str)?;
        match str.as_bytes().get(0) {
            Some(b'^') => {
                *str = &str[1..];
                let sq2 = Square::parse(str)?;
                let sq3 = Square::parse(str)?;
                return Ok(Move::Split{ from: sq1, to_1: sq2, to_2: sq3 });
            }
            Some(b'a'..= b'h') => (),
            Some(_) => return Err("Expected '^' or a square".into()),
            None => return Err("Need two squares in move".into())
        };

        let sq2 = Square::parse(str)?;
        match str.as_bytes().get(0) {
            None => return Ok(Move::Simple{ from: sq1, to: sq2 }),
            Some(b'^') => {
                *str = &str[1..];
                let sq3 = Square::parse(str)?;
                return Ok(Move::Merge{ from_1: sq1, from_2: sq2, to: sq3 });
            }
            // TODO: Other piece promitions
            Some(b'Q') => {
                *str = &str[1..];
                return Ok(Move::Promote{ from: sq1, to: sq2, kind: PieceKind::Queen })
            }
            _ => return Err(format!("Unexpected trailing characters: {}", str))
        }
    }

    fn update_castling(self, team: bool, mut castling: [[bool; 2]; 2]) -> [[bool; 2]; 2] {
        let team_idx = if team { 0 } else { 1 };
        let mut update_from = |sq| match sq {
            Square{ col: Column::E, row} if row.perspective(team) == Row::R1 =>
                castling[team_idx] = [false; 2],
            Square{ col: Column::A, row } if row.perspective(team) == Row::R1 =>
                castling[team_idx][1] = false,
            Square{ col: Column::H, row } if row.perspective(team) == Row::R1 =>
                castling[team_idx][0] = false,
            _ => ()
        };
        match self {
            Move::Simple{from, ..} => update_from(from),
            // Maybe I'm also supposed to update to? That feels unecessary
            Move::Split{from, ..} => update_from(from),
            Move::Merge{from_1, from_2, ..} => {
                update_from(from_1);
                update_from(from_2);
            }
            Move::Promote{..} => (),
        }

        castling
    }
}
