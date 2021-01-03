use qc::{
    get_probs,
    ChessState,
    Row,
    Column,
    Move,
    Square
};

fn state_string(state: &ChessState) -> String {
    use std::fmt::Write;
    // 8 rows * 4 height * 8 columns * 7 width + 8 newlines
    let mut out = String::with_capacity(8 * 4  *  8 * 7  +  8);

    let pieces = state.pieces;
    let probs = get_probs(state);

    out += "  ";
    for i in 0.. 8 * 4 {
        if i % 4 == 2 {
            out.push((b'A' + (i / 4)) as char);
        }
        else {
            out.push(' ');
        }
    }
    out += "\n  ";
    for _ in 0.. 8 * 4 + 1 {
        out += "-";
    }
    out += "\n";
    for row in 0.. 8 {
        // Piece line
        out.push((b'1' + row as u8) as char);
        out.push(' ');
        for col in 0.. 8 {
            let row = Row::from_idx(row).unwrap();
            let col = Column::from_idx(col).unwrap();
            let sq = Square{ row, col };
            let piece = pieces[sq.get_idx()];
            let c = match piece {
                Some(p) => p.symbol(),
                None => ' ',
            };
            write!(out, "| {} ", c).unwrap();
        }
        out += "|\n  ";
        // Prob line
        for col in 0.. 8 {
            let prob = probs[row * 8 + col];
            assert!(prob < 1.001);
            if prob != 0.0 && prob <= 0.999 {
                // let s = format!("{:.2}", prob);
                write!(out, "|.{:^2}", (prob * 100.).round()).unwrap();
            }
            else {
                out += "|   ";
            }
        }
        // End prob line + padding
        out += "|\n  ";
        for _ in 0.. 8 * 4 + 1 {
            out += "-";
        }
        out += "\n";
    }

    out
}

#[cfg(feature = "future")]
fn board_string(pieces: [Option<Piece>; 64]) -> String {
    let mut out = String::with_capacity(64 + 8);
    for row in 0.. 8 {
        for col in 0.. 8 {
            let row = Row::from_idx(row).unwrap();
            let col = Column::from_idx(col).unwrap();
            let sq = Square{ row, col };
            let piece = pieces[sq.get_idx()];
            let c = match piece {
                Some(p) => p.symbol(),
                None => ' ',
            };
            out.push(' ');
            out.push(c);
        }
        out.push('\n');
    }
    out
}

fn fake_main() -> Result<(), std::io::Error> {
    use std::io::{self, BufRead, prelude::*};
    use std::str::FromStr;
    use std::fs::File;

    let seed = u64::from_str(&std::env::args().nth(1).unwrap()).unwrap();
    unsafe {
        extern "C" { fn srand48(x: u64); }
        srand48(seed);
    }

    let mut outfile = File::create("moves_out").unwrap();

    let mut state = ChessState::new_game();
    let input = io::BufReader::new(io::stdin());
    println!("{}", state_string(&state));
    println!("White to start the game");
    for line in input.lines() {
        let mut src: &str = &*(line?);
        writeln!(outfile, "{}", src).unwrap();
        let chess_move = match Move::parse(&mut src) {
            Ok(cm) => cm,
            Err(e) => {
                println!("{}\nTry again", e);
                continue
            }
        };
        match state.apply_move(chess_move) {
            Ok((new_state, Some(m))) => {
                state = new_state;
                println!("Success; m{}", if m {"1"} else {"0"})
            },
            Ok((new_state, None)) => {
                state = new_state;
                println!("Success")
            },
            Err(e) => println!("{}\nTry again", e)
        }

        let next = if state.is_white_turn { "White" } else { "Black" };
        state.occupancy.print_state();
        println!("{}", state_string(&state));
        println!("{} to play", next);
    }

    Ok(())
}

fn main() {
    use std::thread;
    let builder = thread::Builder::new()
                  .name("reductor".into())
                  .stack_size(1024 * 1024 * 1024); // 1GB of stack space

    let handler = builder.spawn(|| {
        fake_main().unwrap()
    }).unwrap();

    handler.join().unwrap();
}