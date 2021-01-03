/// A wrapper around qcomp that tries to help make it
/// "explainable" and tries to expose a cleaner api
/// (with the const related shenanigans hidden)

use std::mem::replace;

use crate::qcomp::{
    simple_log_2,
    BasisVector,
    Complex,
    DynamicMatrix,
    Matrix,
    QuantumSimulator,
    QuantumState,
};

#[derive(Clone, Debug, Default)]
pub struct AllocInfo {
    // TODO: Don't tie this to chess types... make generic?
    pub piece: Option<crate::Piece>,
    name: String,
    purpose: String,
    freed: bool,
}

pub struct SimpleQuantumState {
    pub state: Box<dyn QuantumSimulator>,
    // Todo: Efficient type for cloning?
    pub qbits: Vec<AllocInfo>,
}

impl SimpleQuantumState {
    pub fn new<const N: usize>(bv: BasisVector<N>, mut name_fn: impl FnMut(usize) -> String) -> Self
    where [u8; (N + 7) / 8]:  {
        let mut qbits = Vec::with_capacity(N);
        for i in 0.. N {
            qbits.push(AllocInfo {
                piece: None,
                name: name_fn(i),
                purpose: "".into(),
                freed: false,
            });
        }
        let mut state = QuantumState::<N>::empty();
        state.set(bv, Complex([1.0, 0.0]));
        SimpleQuantumState {
            state: Box::new(state),
            qbits
        }
    }

    pub fn apply_unitary<const UN: usize>(
        &mut self,
        u: Matrix<UN>,
        indicies: [usize; simple_log_2(UN)],
    ) {
        self.print_unitary(u, indicies);

        self.state = self.state.apply_unitary_dynamic(
            DynamicMatrix::new(&u),
            &indicies,
        );
    }

    #[cfg(feature = "future")]
    pub fn apply_unitary_and_clone<const UN: usize>(
        &self,
        u: Matrix<UN>,
        indicies: [usize; simple_log_2(UN)],
    ) -> Self {
        self.print_unitary(u, indicies);

        let state = self.state.apply_unitary_dynamic(
            DynamicMatrix::new(&u),
            &indicies,
        );

        SimpleQuantumState {
            state,
            qbits: self.qbits.clone()
        }
    }

    pub fn apply_controlled_unitaries<const UN: usize, const UNN: usize>(
        &mut self,
        us: [(&Matrix<UN>, Option<&[usize]>, &[usize], &[usize]); UNN],
        indicies: [usize; simple_log_2(UN)]
    ) {
        self.print_controlled_unitaries(us, indicies);

        let dyn_us =
            us.map(|(u, c_any_t, c_all_t, cf)| (DynamicMatrix::new(u), c_any_t, c_all_t, cf));
        self.state = self.state.apply_controlled_unitaries_dynamic(&dyn_us, &indicies);
    }

    pub fn apply_controlled_unitaries_and_clone<const UN: usize, const UNN: usize>(
        &self,
        us: [(&Matrix<UN>, Option<&[usize]>, &[usize], &[usize]); UNN],
        indicies: [usize; simple_log_2(UN)]
    ) -> Self {
        self.print_controlled_unitaries(us, indicies);

        let dyn_us =
            us.map(|(u, c_any_t, c_all_t, cf)| (DynamicMatrix::new(u), c_any_t, c_all_t, cf));
        let state = self.state.apply_controlled_unitaries_dynamic(&dyn_us, &indicies);

        SimpleQuantumState {
            state,
            qbits: self.qbits.clone()
        }
    }

    pub fn measure(&mut self, qbit: usize) -> bool {
        println!("Measuring {} ({})", self.name(qbit), self.qbits[qbit].purpose);
        let (out, prob) = self.state.measure(qbit);
        println!("\tValue: {out}, which happens {:.2}% of the time", prob * 100.0);
        out
    }

    /// Get a new qubit. If no qubit's have been freed and we have
    /// not yet hit our maximum, this returns qubit n where n qubits
    /// have been previously allocated (useful if you want non
    /// ancilla qubits at fixed positions). After qubits have been
    /// freed all bets are off.
    pub fn alloc_qubit(&mut self, piece: Option<crate::Piece>, name: String, purpose: String) -> usize {
        let state = replace(&mut self.state, Box::new(PlaceHolderSimulator));
        let (state, qbit) = state.alloc_qubit();
        self.state = state;
        self.record_alloc_qbit(qbit, piece, name, purpose);
        qbit
    }

    pub fn alloc_qubit_and_clone(&self, piece: Option<crate::Piece>, name: String, purpose: String)  -> (Self, usize) {
        let (state, qbit) = self.state.alloc_qubit_ref();
        let mut new_state = SimpleQuantumState{
            state,
            qbits: self.qbits.clone()
        };
        new_state.record_alloc_qbit(qbit, piece, name, purpose);

        (new_state, qbit)
    }

    /// Tell the implementation that *we* are done with this qubit,
    /// the qubit does not need to be in a |0>/product/... state.
    pub fn free_qubit(&mut self, qbit: usize) {
        self.state.free_qubit(qbit);
        self.qbits[qbit].freed = true;
    }

    pub fn print_state(&self) {
        fn print_state_inner<const N: usize>(state: &QuantumState<N>, qbits: &[AllocInfo])
        where [u8; (N + 7) / 8]: {
            println!("QuantumState<{N}>");
            for i in 64.. N {
                println!("\tqbit {} is {}: {} (capturing {:?})", i, qbits[i].name, qbits[i].purpose, qbits[i].piece);
            }
            println!("\tfreed_qubits: {:?}", state.freed_qubits);
            println!("\treusable_qubits_zeros: {:?}", state.reusable_qubits_zeros);
            println!("\treusable_qubits_ones: {:?}", state.reusable_qubits_ones);
            if state.decomposed.len() > 64 {
                println!("{} states <not printing list due to size>", state.decomposed.len());
            }
            else {
                for (bv, c) in state.iter() {
                    println!("\t{bv:?} {c:?}");
                }
            }
        }

        crate::apply_monomorphized_fn!(&self.state, print_state_inner, &self.qbits);
    }

    fn record_alloc_qbit(&mut self, qbit: usize, piece: Option<crate::Piece>, name: String, purpose: String) {
        let info = AllocInfo{ piece, name, purpose, freed: false };
        if qbit == self.qbits.len() {
            self.qbits.push(info)
        } else {
            self.qbits[qbit] = info;
        }
    }

    fn print_controlled_unitaries<const UN: usize, const UNN: usize>(
        &self,
        us: [(&Matrix<UN>, Option<&[usize]>, &[usize], &[usize]); UNN],
        indicies: [usize; simple_log_2(UN)]
    ) {
        println!("\nApplying a set of {UNN} controlled unitaries");
        println!("\tTo: {}", self.qbit_set_str(&indicies));
        for (mat, control_any_true, control_all_true, control_all_false) in &us {
            println!("\tControlled by:");
            if let Some(control_true) = control_any_true {
                if control_true.len() == 0 {
                    println!("\t\tNever");
                }
                else {
                    println!("\t\tAny of: {}", self.qbit_set_str(control_true));
                }
            }
            if control_all_true.len() != 0 {
                println!("\t\tAll of: {}", self.qbit_set_str(control_all_true));
            }
            if control_all_false.len() != 0 {
                println!("\t\tNone of: {}", self.qbit_set_str(control_all_false));
            }
            println!("\tApplying\n{}", mat_str(**mat, "\t\t"));
        }
    }

    fn print_unitary<const UN: usize>(
        &self,
        u: Matrix<UN>,
        indicies: [usize; simple_log_2(UN)],
    ) {
        println!("\nApplying\n{}\n\tto qubits {}",
            mat_str(u, "\t\t"),
            self.qbit_set_str(&indicies),
        );
    }

    fn name(&self, qbit: usize) -> &str {
        &self.qbits[qbit].name
    }

    fn qbit_set_str(&self, qbits: &[usize]) -> String {
        let mut out = String::new();
        for (i, &bit) in qbits.iter().enumerate() {
            out += self.name(bit);
            if i+1 != qbits.len() {
                out += ", "
            }
        }
        out
    }
}

fn mat_str<const N: usize>(mat: Matrix<N>, prefix: &str) -> String {
    let fmt_mat = mat.map(|elem| format!("{:?} ", elem));
    let mut col_widths = [0; N];
    for (i, col) in fmt_mat.column_iter().enumerate() {
        col_widths[i] = col.iter().map(|x| x.len()).max().unwrap();
    }
    let mut out = String::with_capacity(4*N);
    for i in 0.. N {
        out += prefix;
        for j in 0.. N {
            out += &format!("{0:^1$}", fmt_mat[(i, j)], col_widths[j]);
        }
        out += "\n";
    }
    out
}

#[derive(Debug)]
struct PlaceHolderSimulator;
impl QuantumSimulator for PlaceHolderSimulator {
    fn apply_unitary_dynamic(
        &self,
        _u: DynamicMatrix,
        _indicies: &[usize],
    ) -> Box<dyn QuantumSimulator> {
        unreachable!()
    }
    fn apply_controlled_unitaries_dynamic(
        &self,
        // Matrix, control true, control false.
        _unitaries: &[(
            DynamicMatrix,
            Option<&[usize]>,
            &[usize],
            &[usize],
        )],
        _indicies: &[usize],
    ) -> Box<dyn QuantumSimulator> {
        unreachable!()
    }
    fn measure(&mut self, _qubit: usize) -> (bool, f64) {
        unreachable!()
    }
    fn state_size(&self) -> usize {
        unreachable!()
    }
    fn alloc_qubit(self: Box<Self>) -> (Box<dyn QuantumSimulator>, usize) {
        unreachable!()
    }
    fn alloc_qubit_ref(&self) -> (Box<dyn QuantumSimulator>, usize) {
        unreachable!()
    }
    fn free_qubit(&mut self, _qubit: usize) {
        unreachable!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        unreachable!()
    }
    fn eq_any(&self, _other: &dyn QuantumSimulator) -> bool {
        unreachable!()
    }
}