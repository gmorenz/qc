
use crate::qcomp::Matrix;
use aljabar::*;

fn tensor_product<const M: usize, const N: usize>(mat1: Matrix<M>, mat2: Matrix<N>) -> Matrix<{N * M}> {
    let mut out = Matrix::zero();
    for m_j in 0.. M {
        for m_i in 0.. M {
            for n_j in 0.. N {
                for n_i in 0.. N {
                    out[m_j * N + n_j][m_i * N + n_i] = mat1[m_j][m_i] * mat2[n_j][n_i];
                }
            }
        }
    }
    out
}

/// Logical not of a single qubit.
pub fn pauli_x() -> Matrix<2> {
    matrix_c![
        [ 0.0, 1.0 ],
        [ 1.0, 0.0 ]
    ]
}

pub fn swap() -> Matrix<4> {
    matrix_c![
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]
    ]
}

fn swap3_12() -> Matrix<8> {
    tensor_product(Matrix::<2>::one(), swap())
}

pub fn iswap() -> Matrix<4> {
    matrix_c![
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, (i 1.0), 0.0 ],
        [ 0.0, (i 1.0), 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]
    ]
}

pub fn iswap3_01() -> Matrix<8> {
    tensor_product(Matrix::<2>::one(), iswap())
}

pub fn iswap3_02() -> Matrix<8> {
    swap3_12() * iswap3_01() * swap3_12()
}

/// Swap qbit 0 to qbit 1, then qbit 1 to qbit 2
pub fn iswap3_01_12() -> Matrix<8> {
    let ident: Matrix<2> = Matrix::one();
    tensor_product(iswap(), ident) * tensor_product(ident, iswap())
}

pub fn iswap4_01_23() -> Matrix<16> {
    let ident: Matrix<4> = Matrix::one();
    tensor_product(iswap(), ident) * tensor_product(ident, iswap())
}

pub fn iswap5_43_21_10() -> Matrix<32> {
    tensor_product(Matrix::<8>::one(), iswap()) // 1, 0
    * tensor_product(Matrix::<4>::one(), tensor_product(iswap(), Matrix::<2>::one())) // 2, 1
    * tensor_product(iswap(), Matrix::<8>::one()) // 4, 3
}

/// from, to_1, to_2, to_1 from low to high qubit
pub fn qc_split() -> Matrix<8> {
    use std::f64::consts::FRAC_1_SQRT_2;
    matrix_c![
        [ 1.0,         0.0,             0.0,          0.0,     0.0,         0.0,             0.0,          0.0 ],
        [ 0.0,         0.0,             0.0,          0.0,   (i 1.0),       0.0,             0.0,          0.0 ],
        [ 0.0, (i FRAC_1_SQRT_2), FRAC_1_SQRT_2,      0.0,     0.0,         0.0,             0.0,          0.0 ],
        [ 0.0,         0.0,             0.0,          0.0,     0.0,   (-FRAC_1_SQRT_2), (i FRAC_1_SQRT_2), 0.0 ],
        [ 0.0, (i FRAC_1_SQRT_2), (-FRAC_1_SQRT_2),   0.0,     0.0,         0.0,             0.0,          0.0 ],
        [ 0.0,         0.0,             0.0,          0.0,     0.0,   (i FRAC_1_SQRT_2), (-FRAC_1_SQRT_2), 0.0 ],
        [ 0.0,         0.0,             0.0,        (i 1.0),   0.0,         0.0,             0.0,          0.0 ],
        [ 0.0,         0.0,             0.0,          0.0,     0.0,         0.0,             0.0,          1.0 ]
    ]
}

/// to, from_2, from_1 from low to high qubit
pub fn qc_merge() -> Matrix<8> {
    use std::f64::consts::FRAC_1_SQRT_2;
    matrix_c![
        [ 1.0,   0.0,          0.0,                 0.0,                 0.0,                0.0,          0.0,    0.0 ],
        [ 0.0,   0.0,    (i -FRAC_1_SQRT_2),        0.0,         (i -FRAC_1_SQRT_2),         0.0,          0.0,    0.0 ],
        [ 0.0,   0.0,      FRAC_1_SQRT_2,           0.0,         (-FRAC_1_SQRT_2),           0.0,          0.0,    0.0 ],
        [ 0.0,   0.0,          0.0,                 0.0,                 0.0,                0.0,        (i -1.0), 0.0 ],
        [ 0.0, (i -1.0),       0.0,                 0.0,                 0.0,                0.0,          0.0,    0.0 ],
        [ 0.0,   0.0,          0.0,           (-FRAC_1_SQRT_2),          0.0,        (i -FRAC_1_SQRT_2),   0.0,    0.0 ],
        [ 0.0,   0.0,          0.0,          (i -FRAC_1_SQRT_2),         0.0,         (-FRAC_1_SQRT_2),    0.0,    0.0 ],
        [ 0.0,   0.0,          0.0,                 0.0,                 0.0,               0.0,           0.0,    1.0 ]
    ]
}
