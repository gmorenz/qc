/// Implementation of a quantum computer simulator
///
/// We favor brute force speed over smart algorithms,
/// which makes sense for some problems and not others.
///
/// Only supports quantum states of a fixed set of number
/// of qbits. That set can be modified by modifying the
/// list of numbers in `with_bound_const`. This allows us
/// to get away with using *very* few allocations and while
/// I haven't benchmarked I believe means we should be very
/// fast on reasonably complex states.
///
/// The storage is optimized for pure states with relatively
/// few non-zero computational basis elements. If most
/// computational basis elements are non zero this could be
/// made much faster by replacing
/// `decomposed: HashMap<BasisVector<N>, Complex>`
/// with `decomposed: [Complex; 1 << N]`. It should be easy
/// to put this variation behind a feature flag if anyone is
/// interested in it.
///
/// Has support for allocating and freeing qbits, and soon
/// to have support for reversing computation to previous
/// states. Both of these could be easily removed or moved
/// behind a feature flag for minor performance improvements.
/// Support for freeing qbits is currently sub-optimal (they
/// are only freed if they reach a |0> or |1> state, when they
/// could be freed as soon as they are in a set of freed qbits
/// that is tensor producted with the non-freed state).

use std::{
   collections::HashMap,
   fmt,
   marker::PhantomData,
   ops::{Index, Rem, Sub, Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Not},
};

use aljabar::*;

use crate::c;



// Can't use with_const in apply_monomorphized_fn
// because assigning N to a const makes the compiler
// forget the bounds on N :'(
#[macro_export]
macro_rules! with_bound_const {
    ($name:ident with $ty:ident from $type_builder:tt= $val:expr; $($cval:literal)* $expr:block) => {
        match $val {
            $(
                $cval => {
                    type $ty = $type_builder <$cval>;
                    #[allow(dead_code)]
                    const $name: usize = $cval;
                    $expr
                }
            ),*
            _ => panic!("Unexpected value in with_const")
        }
    };

    ($name:ident with $ty:ident from $type_builder:tt= $val:expr; NQBITS $expr:block) => {
        crate::with_bound_const!($name with $ty from $type_builder = $val;
            64 65 66 67 68 69
            70 71 72 73 74 75 76 77 78 79
            80 81 82 83 84 85 86 87 88 89
            90 91 92 93 94 95 96 97 98 99
            100 101 102 103 104 105 106 107 108 109
            110 111 112 113 114 115 116 117 118 119
            120 121 122 123 124 125 126 127 128 129
            130 131 132 133 134 135 136
        $expr
        )
    }
}

#[macro_export]
macro_rules! apply_monomorphized_fn {
    ($state:expr, $func: expr, $($arg:expr),*) => {{
        let state = $state;
        let n = state.state_size();
        let any = state.as_any();

        crate::with_bound_const!(N with QuantumStateN from QuantumState = n;
            NQBITS
        {
            let state = any.downcast_ref::<QuantumStateN>().unwrap();
            $func(state, $($arg),*)
        })
    }}
}

// Defining some useful types

// Rational number type
// TODO: Make this confirurable with an arbitrary precision option?
// I think I just need rat + 1/nat power.
pub type Real = f64;
pub type Matrix<const N: usize> = aljabar::Matrix<Complex, N, N>;
#[derive(Copy, Clone, PartialEq)]
pub struct Complex(pub [Real; 2]);

mod complex_impl {
    // Adding unecessary modules to allow for code folding?
    // Doesn't sound like me at all!
    use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, Div};
    use std::fmt;
    use aljabar::{Zero, One};
    use super::{Complex, Real};

    impl Complex {
        pub fn norm_squared(self) -> Real {
            self[0] * self[0] + self[1] * self[1]
        }
    }

    impl Zero for Complex {
        fn zero() -> Self {
            Complex([0.0; 2])
        }

        fn is_zero(&self) -> bool {
            self == &Self::zero()
        }
    }

    impl One for Complex {
        fn one() -> Self {
            Complex([1.0, 0.0])
        }

        fn is_one(&self) -> bool {
            self == &Self::one()
        }
    }

    impl Eq for Complex {}

    impl Index<usize> for Complex {
        type Output = f64;
        fn index(&self, idx: usize) -> &f64 {
            &self.0[idx]
        }
    }

    impl IndexMut<usize> for Complex {
        fn index_mut(&mut self, idx: usize) -> &mut f64 {
            &mut self.0[idx]
        }
    }

    impl Add for Complex {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let mut out = Self::zero();
            for i in 0.. 2 {
                out[i] += self.0[i] + rhs[i];
            }
            out
        }
    }

    impl Mul for Complex {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Complex([self[0] * rhs[0] - self[1] * rhs[1], self[0] * rhs[1] + self[1] * rhs[0]])
        }
    }

    impl Sub for Complex {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Complex([self[0] - rhs[0], self[1] - rhs[1]])
        }
    }

    impl Div for Complex {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            if rhs[1] != 0.0 { todo!() }
            Complex([self[0] / rhs[0], self[1] / rhs[0]])
        }
    }

    impl AddAssign for Complex {
        fn add_assign(&mut self, rhs: Self) {
            for i in 0.. 2 {
                self.0[i] += rhs[i];
            }
        }
    }

    impl fmt::Debug for Complex {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            match (self[0] == 0.0, self[1] == 0.0) {
                (true, true) => fmt.write_str("0"),
                (false, true) => fmt.write_fmt(format_args!("{}", self[0])),
                (true, false) => fmt.write_fmt(format_args!("(i {})", self[1])),
                (false, false) => fmt.write_fmt(format_args!("({} + i {})", self[0], self[1]))
            }
        }
    }
}

/// Generally an internel implementation detail, used to allow for
/// object-safe trait implementation.
pub struct DynamicMatrix<'a> {
    n: usize,
    matrix: *const (),
    _marker: PhantomData<&'a ()>
}

impl<'a> DynamicMatrix<'a> {
    pub fn new<const N: usize>(mat: &'a Matrix<N>) -> Self {
        DynamicMatrix {
            n: N,
            matrix: mat as *const _ as *const (),
            _marker: PhantomData,
        }
    }

    fn into_fixed<const N: usize>(&self) -> &Matrix<N> {
        unsafe {
            assert_eq!(self.n, N);
            &*(self.matrix as *const Matrix<N>)
        }
    }
}

// a 64 bit bit vector, convenientally just large enough
// for a chess board
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct BasisVector<const BITS: usize>
where [u8; (BITS + 7) / 8]: {
    // Note that the left over in bytes we aren't using
    // must be set to zero.
    bytes: [u8; (BITS + 7) / 8],
}
impl<const N: usize> From<u64> for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    fn from(x: u64) -> Self {
        use std::convert::TryFrom;
        let x_bytes = x.to_le_bytes();
        let bytes = <[u8; (N + 7) / 8]>::try_from(&x_bytes[0.. (N + 7) / 8]).unwrap();
        Self{ bytes }
    }
}
impl<'a, const N: usize> From<&'a [usize]> for BasisVector<N>
where [u8; (N + 7) / 8]: {
    fn from(indicies: &'a [usize]) -> Self {
        let mut ret = Self::zero();
        for &i in indicies {
            ret.set(i, true);
        }
        ret
    }
}
impl<const N: usize> Index<usize> for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    type Output=bool;
    fn index(&self, idx: usize) -> &bool {
        if (self.bytes[idx / 8] >> (idx % 8)) & 0x1 == 1 {
            &true
        } else { &false }
    }
}
impl<const N: usize> BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    fn find_one(&self) -> Option<usize> {
        for (i, byte) in self.bytes.iter().enumerate() {
            let tz = byte.trailing_zeros();
            if tz != 8 {
                if (i * 8 + tz as usize) > N {
                    panic!("wtf?");
                }
                return Some(i * 8 + tz as usize);
            }
        }
        None
    }

    fn leading_ones(&self) -> usize {
        if N == 0 { return 0 };
        let initial;
        if N % 8 == 0 {
            initial = 0;
        }
        else {
            let last_byte = self.bytes.last().unwrap();
            let tail_len = 8 - (N % 8);
            initial = (last_byte << tail_len).leading_ones() as usize;
            if initial != N % 8 {
                return initial;
            }
        }

        for (i, byte) in self.bytes.iter().rev().skip(1).enumerate() {
            let trail = byte.leading_ones();
            if trail != 8 {
                return initial + i * 8 + trail as usize;
            }
        }

        assert_eq!(N % 8, 0);
        return N
    }

    pub fn set(&mut self, idx: usize, val: bool) {
        let byte = &mut self.bytes[idx / 8];
        let bit = idx % 8;
        if val {
            *byte |= 1 << bit
        } else {
            *byte &= !(1 << bit)
        }
    }

    pub fn zero() -> Self {
        Self{ bytes: [0; (N + 7) / 8] }
    }

    fn max() -> Self {
        let mut bv = Self{ bytes: [!0; (N + 7) / 8] };
        // Zero out extra in last byte if needed
        if N % 8 != 0 {
            let used_bits = N % 8;
            bv.bytes[bv.bytes.len() - 1] &= (!0) >> (8 - used_bits);
        }
        bv
    }

    // Keep private to make it harder to confuse indicies and BasisVectors
    fn from_usize(x: usize) -> Self {
        let mut out = Self::zero();
        let x_bytes = x.to_le_bytes();
        let l = ::std::mem::size_of::<usize>().min((N + 7) / 8);
        out.bytes[0.. l].copy_from_slice(&x_bytes[0.. l]);
        out
    }

    /// Adds |0> qubits on the end, or removes qubits from the end
    pub fn resize<const M: usize>(&self) -> BasisVector<M>
    where [u8; (M + 7) / 8]: ,{
        let mut bv = BasisVector::<M>::zero();
        let l_n = (N + 7) / 8;
        let l_m = (M + 7) / 8;
        for i in 0.. l_n.min(l_m) {
            bv.bytes[i] = self.bytes[i]
        }
        // Zero out extra in last byte if needed
        if N > M && (M % 8) != 0 {
            let used_bits = M % 8;
            bv.bytes[l_m - 1] &= (!0) >> (8 - used_bits);
        }
        bv
    }
}
impl<const N: usize> fmt::Debug for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use std::str;
        // Would make this const... but it complains that the length might overflow
        // and thererfore the type isn't well formed :'(. It's also much
        // simpler to just use a dynamic array, and it's just debugging code, so meh.
        let mut out = Vec::with_capacity(N + N/8);
        for i in 0.. N {
            if i != 0 && i % 8 == 0 {
                out.push(b'_');
            }
            out.push(if self[i] { b'1' } else { b'0' });
        }
        fmt.write_fmt(format_args!("bv[{}]{} {:x?}", N, str::from_utf8(&out).unwrap(), self.bytes))
    }
}
impl<const N: usize> Rem<usize> for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    type Output = usize;
    fn rem(self, rhs: usize) -> usize {
        assert!(rhs.is_power_of_two() && rhs <= 256, "Unsupported modulos for bitvec {}", rhs);
        self.bytes[0] as usize % rhs
    }
}
impl<const N: usize> Sub<usize> for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    type Output = Self;
    fn sub(mut self, rhs: usize) -> Self {
        let rhs = Self::from_usize(rhs);
        let mut carry: u8 = 0;
        for idx in 0.. (N + 7) / 8 {
            if rhs.bytes[idx] == !0 && carry == 1 {
                // Subtract 256, i.e. do nothing to this byte and carry remains 1.
                continue
            }
            let sub = rhs.bytes[idx] + carry;
            let (r, overflow) = self.bytes[idx].overflowing_sub(sub);
            self.bytes[idx] = r;
            carry = if overflow { 1 } else { 0 };
        }

        if carry == 1 {
            panic!("Subtraction undeflow")
        }

        self
    }
}
impl<const N: usize> Add<usize> for BasisVector<N>
where [u8; (N + 7) / 8]: ,{
    type Output = Self;
    fn add(mut self, rhs: usize) -> Self {
        let rhs = Self::from_usize(rhs);
        let mut carry = 0;
        for idx in 0.. (N + 7) / 8 {
            let r = self.bytes[idx] as u32 + rhs.bytes[idx] as u32 + carry;
            self.bytes[idx] = (r % 256) as u8;
            carry = r / 256;
        }
        self
    }
}
impl<const N: usize> BitAndAssign for BasisVector<N>
where [u8; (N + 7) / 8]: {
    fn bitand_assign(&mut self, rhs: Self) {
        for idx in 0.. (N + 7) / 8 {
            self.bytes[idx] &= rhs.bytes[idx]
        }
    }
}
impl<const N: usize> BitAnd for BasisVector<N>
where [u8; (N + 7) / 8]: {
    type Output = Self;
    fn bitand(mut self, rhs: Self) -> Self {
        self &= rhs;
        self
    }
}
impl<const N: usize> BitOrAssign for BasisVector<N>
where [u8; (N + 7) / 8]: {
    fn bitor_assign(&mut self, rhs: Self) {
        for idx in 0.. (N + 7) / 8 {
            self.bytes[idx] |= rhs.bytes[idx]
        }
    }
}
impl<const N: usize> BitOr for BasisVector<N>
where [u8; (N + 7) / 8]: {
    type Output = Self;
    fn bitor(mut self, rhs: Self) -> Self {
        self |= rhs;
        self
    }
}
impl<const N: usize> Not for BasisVector<N>
where [u8; (N + 7) / 8]: {
    type Output = Self;
    fn not(mut self) -> Self {
        for byte in &mut self.bytes {
            *byte = !*byte;
        }
        // Zero out extra in last byte if needed
        if N % 8 != 0 {
            let used_bits = N % 8;
            self.bytes[self.bytes.len() - 1] &= (!0) >> (8 - used_bits);
        }
        self
    }
}

/// N QuBit quantum state
#[derive(Clone, Debug)]
pub struct QuantumState<const N: usize>
where [u8; (N + 7) / 8]: ,{
    /// Fully expanded, this would be 2^64 complex numbers...
    /// Very likely want to consider using im for this!
    pub decomposed: HashMap<BasisVector<N>, Complex>,

    /// Qubits which have been freed (but may still be entangled)
    pub freed_qubits: BasisVector<N>,
    /// Qubits which have been freed and are no longer in use, in a
    /// |0> state.
    pub reusable_qubits_zeros: BasisVector<N>,
    /// Qubits which have been freed and are no longer in use, in a
    /// |1> state.
    pub reusable_qubits_ones: BasisVector<N>,
}

impl<const N: usize> PartialEq for QuantumState<N>
where [u8; (N + 7) / 8]: {
    fn eq(&self, rhs: &QuantumState<N>) -> bool {
        self.decomposed == rhs.decomposed
    }
}


/// Only works on powers of 2
pub(crate) const fn simple_log_2(x: usize) -> usize {
    assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

pub trait QuantumSimulator: std::fmt::Debug + std::any::Any {
    fn apply_unitary_dynamic(
        &self,
        u: DynamicMatrix,
        indicies: &[usize],
    ) -> Box<dyn QuantumSimulator>;

    fn apply_controlled_unitaries_dynamic(
        &self,
        // Matrix, control true, control false.
        unitaries: &[(
            DynamicMatrix,
            Option<&[usize]>,
            &[usize],
            &[usize],
        )],
        indicies: &[usize],
    ) -> Box<dyn QuantumSimulator>;

    fn measure(&mut self, qubit: usize) -> (bool, Real);

    fn state_size(&self) -> usize;

    /// Get a new qubit. If no qubit's have been freed and we have
    /// not yet hit our maximum, this returns qubit n where n qubits
    /// have been previously allocated (useful if you want non
    /// ancilla qubits at fixed positions). After qubits have been
    /// freed all bets are off.
    fn alloc_qubit(self: Box<Self>) -> (Box<dyn QuantumSimulator>, usize);
    fn alloc_qubit_ref(&self) -> (Box<dyn QuantumSimulator>, usize);
    /// Tell the implementation that *we* are done with this qubit,
    /// the qubit does not need to be in a |0>/product/... state.
    fn free_qubit(&mut self, qubit: usize);
    // fn run_gc(self: Box<Self>) -> (Box<dyn QuantumSimulator>);

    #[doc(hidden)]
    fn as_any(&self) -> &dyn std::any::Any;
    #[doc(hidden)]
    fn eq_any(&self, other: &dyn QuantumSimulator) -> bool;
}

impl PartialEq for dyn QuantumSimulator {
    fn eq(&self, other: &Self) -> bool {
        self.eq_any(other)
    }
}

impl<const N: usize> QuantumSimulator for QuantumState<N>
where [u8; (N + 7) / 8]: {
    fn apply_unitary_dynamic(
        &self,
        u: DynamicMatrix,
        indicies: &[usize],
    ) -> Box<dyn QuantumSimulator> {
        use std::convert::TryFrom;
        // Too big for the stack at big sizes!
        // Unfortunate that this trick means we aren't inlining matrix values anymore...
        // but probably impossible.
        crate::with_const!(N: usize = u.n; 2 4 8 16 32 64 128 256 1024 {
            assert_eq!(indicies.len(), simple_log_2(N));
            let matrix = u.into_fixed::<N>();
            let indicies = <[usize; simple_log_2(N)]>::try_from(indicies).unwrap();
            self.apply_fixed_unitary(matrix, indicies)
        })
    }

    fn apply_controlled_unitaries_dynamic(
        &self,
        unitaries: &[(
            DynamicMatrix,
            Option<&[usize]>,
            &[usize],
            &[usize],
        )],
        indicies: &[usize]
    ) -> Box<dyn QuantumSimulator> {
        use std::convert::TryFrom;
        crate::with_const!(LOG2_UN: usize = indicies.len(); 1 2 3 4 5 {
            let indicies = <[usize; LOG2_UN]>::try_from(indicies).unwrap();

            crate::with_const!(UNN: usize = unitaries.len();1 2 3 4 5 {
                let mut fixed_us = [ControlledUnitary::<N, {1 << LOG2_UN}>::zero(); UNN];
                for i in 0.. UNN {
                    assert_eq!(unitaries[i].0.n, 1 << LOG2_UN);
                    fixed_us[i].matrix = *unitaries[i].0.into_fixed::<{1 << LOG2_UN}>();
                    if let Some(c_true) = unitaries[i].1 {
                        fixed_us[i].control_any_true = Some(c_true.into());
                    }
                    fixed_us[i].control_all_true = unitaries[i].2.into();
                    fixed_us[i].control_all_false = unitaries[i].3.into();
                }

                self.apply_controlled_unitaries_fixed(
                    fixed_us,
                    indicies
                )
            })
        })
    }

    fn measure(&mut self, idx: usize) -> (bool, Real) {
        let mut prob_true = 0.0;
        for (basis, v) in self.decomposed.iter() {
            if basis[idx] {
                prob_true += v.norm_squared()
            }
        }

        let measure  = prob_true >= rand();
        let prob = if measure { prob_true } else { 1.0 - prob_true };
        self.decomposed.drain_filter(|basis, val| {
            // Renormalize
            *val = *val / c!(prob.sqrt());
            basis[idx] != measure
        }).for_each(drop);

        self.update_reusability();
        (measure, prob)
    }

    fn state_size(&self) -> usize { N }

    fn alloc_qubit_ref(&self) -> (Box<dyn QuantumSimulator>, usize) {
        if let Some(idx) = self.reusable_qubits_zeros.find_one() {
            let mut out = self.clone();
            out.reusable_qubits_zeros.set(idx, false);
            return (Box::new(out), idx)
        }

        if let Some(idx) = self.reusable_qubits_ones.find_one() {
            // TODO: alloc_qubit_ref_mut to avoid cloning?
            let mut out = self.clone();
            out.reusable_qubits_ones.set(idx, false);
            // Set qubit to zero
            let new_iter = self.iter().map(|(mut bv, c)| {
                bv.set(idx, false);
                (bv, c)
            });
            out.decomposed = new_iter.collect();
            return (Box::new(out), idx);
        }


        // Roughly *half* my compile time is these 4 lines of code, because it
        // generates code sized quadratic in the number of supported qbits.
        // I haven't been able to figure out how to avoid this, because I"m not allowed
        // to do "N+1" because "that might panic" in types.
        crate::with_bound_const!(M with QuantumStateM from QuantumState = N+1; NQBITS {
            let out: QuantumStateM = self.resize();
            (Box::new(out), N)
        })
    }

    fn alloc_qubit(mut self: Box<Self>) -> (Box<dyn QuantumSimulator>, usize) {
        if let Some(idx) = self.reusable_qubits_zeros.find_one() {
            self.reusable_qubits_zeros.set(idx, false);
            return (self, idx)
        }

        self.alloc_qubit_ref()
    }

    fn free_qubit(&mut self, qubit: usize) {
        self.freed_qubits.set(qubit, true);
        self.update_reusability();
    }

    // fn run_gc(self: Box<Self>) -> (Box<dyn QuantumSimulator>) {
    //     todo!()
    // }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn eq_any(&self, other: &dyn QuantumSimulator) -> bool {
        // Quadratically many functions in number of qubits supported...
        // probably not ideal...
        fn is_eq<const N: usize, const M: usize>(lhs: &QuantumState<N>, rhs: &QuantumState<M>) -> bool
        where [u8; (M + 7) / 8]: ,[u8; (N + 7) / 8]: {
            assert!(N >= M);

            // Check that we don't use state past M
            if N != M {
                let reusable = lhs.reusable_qubits_ones | lhs.reusable_qubits_zeros;
                if N - reusable.leading_ones() > M {
                    return false
                }
            }

            for (bv, val) in lhs.iter() {
                if rhs.decomposed.get(&bv.resize()) != Some(&val) {
                    return false;
                }
            }

            true
        }

        fn is_eq_unsorted<const N: usize, const M: usize>(lhs: &QuantumState<N>, rhs: &QuantumState<M>) -> bool
        where [u8; (M + 7) / 8]: ,[u8; (N + 7) / 8]: {
            if N > M { is_eq(lhs, rhs) }
            else { is_eq(rhs, lhs) }
        }

        apply_monomorphized_fn!(other, is_eq_unsorted, self)
    }
}

fn make_swap_list<const N: usize, const IN: usize>(indicies: [usize; IN]) -> [(usize, usize); IN] {
    // Create a list of swap that we need to perform to move the qubits
    // listed in indicies to the front of the state.
    let mut index_map: [usize; N] = [0; N];
    let mut swap_list = [(0, 0); IN];
    for i in 0.. N {
        index_map[i] = i;
    }
    for (i, &idx) in indicies.iter().enumerate() {
        // Find idx in the list.
        let mut current_idx_idx = idx;
        while index_map[current_idx_idx] != idx {
            current_idx_idx = index_map[current_idx_idx];
        }
        // Swap "i" and "current_idx_idx"
        let current_i = index_map[i];
        index_map[i] = idx; // idx = index_map[current_idx_idx]
        index_map[current_idx_idx] = current_i;
        swap_list[i] = (i, current_idx_idx);
    }

    swap_list
}

#[derive(Copy, Clone)]
struct ControlledUnitary<const N: usize, const UN: usize>
where [u8; (N + 7) / 8]: {
    matrix: Matrix<UN>,
    /// Any of these bits
    control_any_true: Option<BasisVector<N>>,
    control_all_true: BasisVector<N>,
    /// All of these bits
    control_all_false: BasisVector<N>,
}

impl<const N: usize, const UN: usize> ControlledUnitary<N, UN>
where [u8; (N + 7) / 8]: {
    fn zero() -> Self {
        ControlledUnitary {
            matrix: Matrix::zero(),
            control_any_true: None,
            control_all_true: BasisVector::zero(),
            control_all_false: BasisVector::zero(),
        }
    }
}

impl<const N: usize> QuantumState<N>
where [u8; (N + 7) / 8]: {
    /// Not actually a valid state!
    pub fn empty() -> QuantumState<N> {
        QuantumState {
            decomposed: HashMap::new(),
            freed_qubits: BasisVector::zero(),
            reusable_qubits_ones: BasisVector::zero(),
            reusable_qubits_zeros: BasisVector::zero(),
        }
    }

    pub fn child<const M: usize>(&self, iter: impl Iterator<Item=(BasisVector<M>, Complex)>) -> QuantumState<M>
    where [u8; (M + 7) / 8]: {
        let mut ret = QuantumState{
            decomposed: HashMap::new(),
            freed_qubits: self.freed_qubits.resize::<M>(),
            reusable_qubits_ones: self.reusable_qubits_ones.resize::<M>(),
            reusable_qubits_zeros: self.reusable_qubits_zeros.resize::<M>(),
        };
        for (bv, val) in iter {
            ret.add(bv, val)
        }
        ret
    }

    fn update_reusability(&mut self) {
        let mut all_ones = BasisVector::max();
        let mut all_zeros = BasisVector::zero();
        for (bv, _) in self.iter() {
            all_ones &= bv;
            all_zeros |= bv;
        }

        let freed_zeros = self.freed_qubits & !all_zeros;
        let freed_ones = self.freed_qubits & all_ones;
        self.freed_qubits &= !(freed_zeros | freed_ones);
        self.reusable_qubits_zeros |= freed_zeros;
        self.reusable_qubits_ones |= freed_ones;
    }

    fn get(&self, idx: BasisVector<N>) -> Complex {
        self.decomposed.get(&idx).copied().unwrap_or(Complex::zero())
    }

    /// Note: This denormalizes
    pub fn set(&mut self, idx: BasisVector<N>, val: Complex) {
        if val == Complex::zero() {
            self.decomposed.remove(&idx);
        }
        else {
            self.decomposed.insert(idx, val);
        }
    }

    /// Note: Does not preserve normalization
    pub fn add(&mut self, idx: BasisVector<N>, val: Complex) {
        if val == Complex::zero() { return };
        let new_val = self.get(idx) + val;
        self.set(idx, new_val);
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item=(BasisVector<N>, Complex)> + 'a {
        self.decomposed.iter().map(|(&bv, &val)| (bv, val))
    }

    /// Indicies in order of low to high bits (when using them to index matrix cols)
    fn apply_controlled_unitaries_fixed<const UN: usize, const UNN: usize>(
        &self,
        unitaries: [ControlledUnitary<N, UN>; UNN],
        indicies: [usize; simple_log_2(UN)],
    ) -> Box<dyn QuantumSimulator> {
        // TOOD: Make sure multiple unitaries should never be applied simultaneously...

        // TODO: Deduplicate with apply_fixed_unitary

        let swap_list = make_swap_list::<N, {simple_log_2(UN)}>(indicies);

        // Create an iterator with those swaps applied
        let swapped_iter = self.iter().map(|(pre_bv, val)| {
            let mut bv = pre_bv;
            for &(i, j) in &swap_list {
                let bits_xor = bv[i] ^ bv[j];
                bv.set(i, bits_xor ^ bv[i]);
                bv.set(j, bits_xor ^ bv[j]);
            }
            (pre_bv, bv, val)
        });

        // Apply the unitary to every element of that iterartor
        // which passes the control test (just return the elements that fail)
        let applied_iter = maybe_flat_map(swapped_iter, |(pre_bv, bv, val)| {
            for unitary in &unitaries {
                if let Some(control_true) = unitary.control_any_true {
                    if pre_bv & control_true == BasisVector::zero() {
                        continue;
                    }
                }
                if (pre_bv & unitary.control_all_false) != BasisVector::zero() {
                    continue;
                }
                if (pre_bv & unitary.control_all_true) != unitary.control_all_true {
                    continue;
                }

                // Apply this unitary

                // Multiply bv by u <tensor product> identity

                // The basis vector picks out a column in that unitary.
                // The column is of the form [0 block][u column][0 block].
                // So we need to return a new iterator of the basis vectors
                // for u_column * val_u_column * val.
                let index_column = bv % UN;
                let base_basis_vector = bv - index_column;
                let u = unitary.matrix;
                // The problem is just that the split indicies are out of order.
                // TODO: For tomorrow, audit all matricies and index orders and controls...
                return Ok(u[index_column].into_iter().enumerate().map(move |(i_col, val_u_col)| {
                    let bv_col = (base_basis_vector + i_col).into();
                    let val = val_u_col * val;
                    (bv_col, val)
                }))
            }

            // None of the matricies control bits like this, just emit the original bv.
            return Err((bv, val));
        });

        // Swap the basis elements back
        let output_iter = applied_iter.map(|(mut bv, val): (BasisVector<N>, _)| {
            for &(i, j) in swap_list.iter().rev() {
                let bits_xor = bv[i] ^ bv[j];
                bv.set(i, bits_xor ^ bv[i]);
                bv.set(j, bits_xor ^ bv[j]);
            }
            (bv, val)
        });

        // Sum up the elements of the iterator
        Box::new(self.child(output_iter))
    }

    /// Indicies in order of low to high bits (when using them to index matrix cols)
    pub fn apply_fixed_unitary<const UN: usize>(
        &self,
        u: &Matrix<UN>,
        indicies: [usize; simple_log_2(UN)],
    ) -> Box<dyn QuantumSimulator> {
        let swap_list = make_swap_list::<N, {simple_log_2(UN)}>(indicies);

        // Create an iterator with those swaps applied
        let swapped_iter = self.iter().map(|(pre_bv, val)| {
            let mut bv = pre_bv;
            for &(i, j) in &swap_list {
                let bits_xor = bv[i] ^ bv[j];
                bv.set(i, bits_xor ^ bv[i]);
                bv.set(j, bits_xor ^ bv[j]);
            }
            (bv, val)
        });

        // Apply the unitary to every element of that iterartor
        // which passes the control test (just return the elements that fail)
        let applied_iter = swapped_iter.flat_map(|(bv, val)| {
            // Multiply bv by u <tensor product> identity

            // The basis vector picks out a column in that unitary.
            // The column is of the form [0 block][u column][0 block].
            // So we need to return a new iterator of the basis vectors
            // for u_column * val_u_column * val.
            let index_column = bv % UN;
            let base_basis_vector = bv - index_column;
            u[index_column].into_iter().enumerate().map(move |(i_col, val_u_col)| {
                let bv_col = (base_basis_vector + i_col).into();
                let val = val_u_col * val;
                (bv_col, val)
            })
        });

        // Swap the basis elements back
        let output_iter = applied_iter.map(|(mut bv, val): (BasisVector<N>, _)| {
            for &(i, j) in swap_list.iter().rev() {
                let bits_xor = bv[i] ^ bv[j];
                bv.set(i, bits_xor ^ bv[i]);
                bv.set(j, bits_xor ^ bv[j]);
            }
            (bv, val)
        });

        Box::new(self.child(output_iter))
    }

    #[cfg(feature = "future")]
    /// Split basis vectors into two sets, measure to decide which
    /// set to keep, and throw out the other set.
    fn measure_split(&mut self, partition: impl Fn(BasisVector<N>) -> bool) -> bool {
        let prob_true: f64 = self
            .decomposed
            .iter()
            .filter(|&(&bv, _): &(&BasisVector<N>,&Complex)| partition(bv))
            .map(|(_, val)| val.norm_squared())
            .sum();

        let measure = prob_true >= rand();
        let prob = if measure { prob_true } else { 1.0 - prob_true };
        self.decomposed.drain_filter(|basis, val| {
            *val = *val / c!(prob.sqrt());
            // TODO: Sign?
            partition(*basis) != measure
        }).for_each(drop);

        self.update_reusability();
        measure
    }

    // I'd have seperate "add qubit" and "remove qubit"
    // functions, but I couldn't make const generics happy
    // in it's current state, so instead this is a shared
    // function that can go in either direction.
    //
    // Adds |0> qubits on the end, or removes qubits from
    // the end... err... I'm not sure the removal procedure
    // makes much sense right now because of interference...
    // (Also I'm doing this when tired, maybe it does, or
    // maybe none of this makes sense)
    fn resize<const M: usize>(&self) -> QuantumState<M>
    where [u8; (M + 7) / 8]:, {
        let out_iter = self
            .iter()
            .map(|(bv, val)| (bv.resize::<M>(), val));
        self.child(out_iter)
    }

    #[cfg(feature = "future")]
    fn renormalize(&mut self) {
        let prob: f64 = self.decomposed.values().map(|v| v.norm_squared()).sum();
        self.decomposed.values_mut().for_each(|v| { *v = *v / c!(prob); });
    }
}

fn rand() -> f64{
    extern "C" {
        fn drand48() -> f64;
    }
    unsafe{ drand48() }
}

#[macro_export]
macro_rules! c {
    ((i $val:expr)) => ($crate::c!(i $val));
    (i $val:expr) => ($crate::qcomp::Complex([0.0, $val]));
    (($real:expr, i $imag:expr)) => ($crate::c!($real, i $imag));
    ($real:expr, i $imag:expr) => ($crate::qcomp::Complex([$real, $imag]));
    ($val:expr) => ($crate::qcomp::Complex([$val, 0.0]));
}

macro_rules! matrix_c {
    ($([$($tks:tt),*]),*) => {
        matrix![
            $(
                [
                    $(c!($tks),)*
                ],
            )*
        ]
    }
}

#[cfg(test)]
macro_rules! state {
    ($n:expr, $($basis_vec:expr => $val:expr),*) => {{
        let mut state = $crate::qcomp::QuantumState::<$n>::empty();
        $(
            state.set($basis_vec.into(), $val);
        )*
        Box::new(state) as Box<dyn QuantumSimulator>
    }}
}

fn maybe_flat_map<I, O, OI>(
    mut iter: impl Iterator<Item = I>,
    mut f: impl FnMut(I) -> Result<OI, O>
) -> impl Iterator<Item = O>
where OI: Iterator<Item = O> {
    fn apply<I, O, OI>(
        maybe_oi: &mut Option<OI>,
        iter: &mut impl Iterator<Item = I>,
        f: &mut impl FnMut(I) -> Result<OI, O>
    ) -> Option<O>
    where OI: Iterator<Item = O> {
        // If there's an active iterator returned by an Ok(inner_iter), get the next elem
        // from that.
        if let Some(item) = maybe_oi.as_mut().and_then(|oi| oi.next()) {
            return Some(item)
        } else { *maybe_oi = None }

        // Otherwise either get the next elem, or the next active iterator
        match f(iter.next()?) {
            Ok(oi) => {
                *maybe_oi = Some(oi);
                apply(maybe_oi, iter, f)
            }
            Err(o) => Some(o)
        }
    }

    let mut maybe_oi = None;
    std::iter::from_fn(move || apply(&mut maybe_oi, &mut iter, &mut f))
}

#[test]
fn test_find_one() {
    let bv: BasisVector<15> = BasisVector::from(0b10u64);
    assert_eq!(bv.find_one(), Some(1));
    let bv: BasisVector<15> = BasisVector::from(0b10_0000_0000u64);
    assert_eq!(bv.find_one(), Some(9));
}
#[test]
fn test_leading_ones() {
    fn slow_leading_ones<const N: usize>(bv: BasisVector<N>) -> usize
    where [u8; (N + 7)/8]: {
        let mut out = 0;
        for i in 0.. N {
            if bv[N - i - 1] == false {
                break;
            }
            out += 1;
        }
        return out;
    }
    let bv: BasisVector<15> = BasisVector::from(0b111_1000_1111_1111u64);
    assert_eq!(bv.leading_ones(), slow_leading_ones(bv), "{bv:?}");
    assert_eq!(slow_leading_ones(bv), 4);

    let bv: BasisVector<25>  = BasisVector::from(0x1fffe00);
    assert_eq!(bv.leading_ones(), slow_leading_ones(bv));
    assert_eq!(slow_leading_ones(bv), 16);
}