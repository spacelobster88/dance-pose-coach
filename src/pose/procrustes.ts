// Procrustes / Kabsch 3D alignment for viewpoint-invariant pose comparison.
//
// Two different CAMERA ANGLES rotate the same physical pose into incomparable
// 2D projections. When a 3D detector (BlazePose, eng-B1) provides world
// landmarks, we can undo the viewpoint difference directly in 3D: center both
// poses (cancel translation), then find the single rigid ROTATION (Kabsch) that
// best maps one onto the other (optionally with a uniform SCALE), and compare in
// that aligned frame. A pose seen from angle A and the same pose from angle B
// then differ only by a rotation we have removed, so they compare as identical.
//
// Dependency-free: a small, self-contained 3x3 Kabsch using the Jacobi
// eigensolver on the 3x3 cross-covariance Gram matrices (a 3x3 SVD without
// pulling in a linear-algebra library).

/** A 3D point with a validity flag (mirrors NormalizedPose's 2D points). */
export interface Point3 {
  x: number;
  y: number;
  z: number;
  valid: boolean;
}

export type Mat3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

const IDENTITY3: Mat3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function centroid(pts: Array<{ x: number; y: number; z: number }>): {
  x: number;
  y: number;
  z: number;
} {
  let x = 0,
    y = 0,
    z = 0;
  for (const p of pts) {
    x += p.x;
    y += p.y;
    z += p.z;
  }
  const n = pts.length || 1;
  return { x: x / n, y: y / n, z: z / n };
}

/** Multiply 3x3 matrices: returns A * B. */
function matMul(a: Mat3, b: Mat3): Mat3 {
  const r: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i][k] * b[k][j];
      r[i][j] = s;
    }
  }
  return r;
}

function transpose(a: Mat3): Mat3 {
  return [
    [a[0][0], a[1][0], a[2][0]],
    [a[0][1], a[1][1], a[2][1]],
    [a[0][2], a[1][2], a[2][2]],
  ];
}

function det3(a: Mat3): number {
  return (
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  );
}

/** Apply a 3x3 rotation to a vector. */
function applyMat(m: Mat3, v: { x: number; y: number; z: number }): {
  x: number;
  y: number;
  z: number;
} {
  return {
    x: m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
    y: m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
    z: m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
  };
}

/**
 * Symmetric-eigen decomposition of a 3x3 matrix via cyclic Jacobi rotations.
 * Returns eigenvalues and a matrix V whose COLUMNS are the eigenvectors, with
 * A = V * diag(eigvals) * V^T. Used to build an SVD of a general 3x3 from the
 * eigendecompositions of A^T A.
 */
function jacobiEigen(input: Mat3): { values: [number, number, number]; vectors: Mat3 } {
  // Work on a mutable copy.
  const a: number[][] = input.map((row) => row.slice());
  const v: number[][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let sweep = 0; sweep < 100; sweep++) {
    // Sum of off-diagonal magnitudes; stop when negligible.
    const off =
      Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-14) break;

    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);

        // Rotate rows/cols p,q of A.
        for (let i = 0; i < 3; i++) {
          const aip = a[i][p];
          const aiq = a[i][q];
          a[i][p] = c * aip - s * aiq;
          a[i][q] = s * aip + c * aiq;
        }
        for (let i = 0; i < 3; i++) {
          const api = a[p][i];
          const aqi = a[q][i];
          a[p][i] = c * api - s * aqi;
          a[q][i] = s * api + c * aqi;
        }
        // Accumulate the rotation into V.
        for (let i = 0; i < 3; i++) {
          const vip = v[i][p];
          const viq = v[i][q];
          v[i][p] = c * vip - s * viq;
          v[i][q] = s * vip + c * viq;
        }
      }
    }
  }

  return {
    values: [a[0][0], a[1][1], a[2][2]],
    vectors: [
      [v[0][0], v[0][1], v[0][2]],
      [v[1][0], v[1][1], v[1][2]],
      [v[2][0], v[2][1], v[2][2]],
    ],
  };
}

/** Column i of a 3x3 matrix as a vector. */
function col(m: Mat3, i: number): { x: number; y: number; z: number } {
  return { x: m[0][i], y: m[1][i], z: m[2][i] };
}

function setCol(m: Mat3, i: number, v: { x: number; y: number; z: number }): void {
  m[0][i] = v.x;
  m[1][i] = v.y;
  m[2][i] = v.z;
}

function vlen(v: { x: number; y: number; z: number }): number {
  return Math.hypot(v.x, v.y, v.z);
}

function cross(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number },
): { x: number; y: number; z: number } {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

/**
 * Optimal proper rotation (det = +1) that best maps `from` onto `to`, both
 * given as matched, already-centered point sets, such that `to ≈ R * from`.
 *
 * Classic Kabsch via the SVD of the 3x3 cross-covariance
 *   H = Σ_k from_k ⊗ to_k        (so that R = V * U^T maps from → to).
 *
 * The SVD is built robustly from ONE symmetric eigendecomposition: eigenvectors
 * V of H^T H (sorted by descending eigenvalue) are the right singular vectors;
 * the matching left singular vectors are U[:,i] = H·V[:,i] / σ_i. A vanishing σ_i
 * (planar / rank-deficient data) is handled by completing U (and V) into a
 * right-handed orthonormal basis via a cross product, which keeps R a valid
 * rotation. A final det check flips the least-significant axis to forbid a
 * reflection.
 */
export function kabschRotation(
  from: Array<{ x: number; y: number; z: number }>,
  to: Array<{ x: number; y: number; z: number }>,
): Mat3 {
  const n = Math.min(from.length, to.length);
  if (n === 0) return IDENTITY3.map((r) => r.slice()) as Mat3;

  // Cross-covariance H[i][j] = Σ_k from_k[i] * to_k[j].
  const H: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let k = 0; k < n; k++) {
    const f = [from[k].x, from[k].y, from[k].z];
    const t = [to[k].x, to[k].y, to[k].z];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        H[i][j] += f[i] * t[j];
      }
    }
  }

  // Right singular vectors V = eigenvectors of H^T H, sorted by descending σ².
  const eig = jacobiEigen(matMul(transpose(H), H));
  const order = [0, 1, 2].sort((a, b) => eig.values[b] - eig.values[a]);
  const V: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let c = 0; c < 3; c++) setCol(V, c, col(eig.vectors, order[c]));

  // Left singular vectors U[:,i] = H·V[:,i] / σ_i; complete degenerate axes.
  const U: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const EPS = 1e-9;
  for (let i = 0; i < 3; i++) {
    const u = applyMat(H, col(V, i));
    const len = vlen(u);
    if (len > EPS) {
      setCol(U, i, { x: u.x / len, y: u.y / len, z: u.z / len });
    } else {
      // σ_i ≈ 0: pick a unit vector orthogonal to the U columns already set.
      let candidate: { x: number; y: number; z: number };
      if (i === 2) {
        candidate = cross(col(U, 0), col(U, 1));
      } else if (i === 1) {
        // Orthogonalize an arbitrary axis against U[:,0].
        const u0 = col(U, 0);
        const seed =
          Math.abs(u0.x) < 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
        const d = seed.x * u0.x + seed.y * u0.y + seed.z * u0.z;
        candidate = { x: seed.x - d * u0.x, y: seed.y - d * u0.y, z: seed.z - d * u0.z };
      } else {
        candidate = { x: 1, y: 0, z: 0 };
      }
      const cl = vlen(candidate) || 1;
      setCol(U, i, { x: candidate.x / cl, y: candidate.y / cl, z: candidate.z / cl });
    }
  }

  // R = V * U^T maps `from` → `to`.
  let R = matMul(V, transpose(U));

  // Forbid a reflection: if det(R) < 0 flip the least-significant left axis.
  if (det3(R) < 0) {
    setCol(U, 2, { x: -U[0][2], y: -U[1][2], z: -U[2][2] });
    R = matMul(V, transpose(U));
  }

  return R;
}

/** Result of aligning a test pose onto a reference pose in 3D. */
export interface AlignedPose3 {
  /** Test points expressed in the reference's aligned frame (centered + rotated [+scaled]). */
  points: Point3[];
  /** The rotation applied to the centered test points. */
  rotation: Mat3;
  /** The uniform scale applied (1 when `scale` was not requested). */
  scale: number;
}

/**
 * Procrustes-align a `test` 3D pose onto a `ref` 3D pose. Both arrays are the
 * full keypoint list (same index order); only entries valid in BOTH poses drive
 * the fit, but every test point is transformed so downstream bone features stay
 * index-aligned.
 *
 * Steps: center each pose on the centroid of its shared valid points, find the
 * optimal Kabsch rotation mapping centered-test → centered-ref over the shared
 * set, optionally a uniform scale, and return the transformed test points in the
 * reference (centered) frame.
 *
 * @param withScale when true also removes a uniform scale difference.
 */
export function procrustesAlign(
  ref: Point3[],
  test: Point3[],
  withScale = false,
): AlignedPose3 | null {
  const n = Math.min(ref.length, test.length);
  const refShared: Array<{ x: number; y: number; z: number }> = [];
  const testShared: Array<{ x: number; y: number; z: number }> = [];
  for (let i = 0; i < n; i++) {
    if (ref[i] && test[i] && ref[i].valid && test[i].valid) {
      refShared.push({ x: ref[i].x, y: ref[i].y, z: ref[i].z });
      testShared.push({ x: test[i].x, y: test[i].y, z: test[i].z });
    }
  }
  // Need at least 3 non-degenerate correspondences to pin a 3D rotation.
  if (refShared.length < 3) return null;

  const refC = centroid(refShared);
  const testC = centroid(testShared);

  const refCentered = refShared.map((p) => ({
    x: p.x - refC.x,
    y: p.y - refC.y,
    z: p.z - refC.z,
  }));
  const testCentered = testShared.map((p) => ({
    x: p.x - testC.x,
    y: p.y - testC.y,
    z: p.z - testC.z,
  }));

  const R = kabschRotation(testCentered, refCentered);

  let scale = 1;
  if (withScale) {
    // Optimal uniform scale s = <R*test, ref> / <test, test> over shared points.
    let num = 0;
    let den = 0;
    for (let k = 0; k < testCentered.length; k++) {
      const rt = applyMat(R, testCentered[k]);
      const rc = refCentered[k];
      num += rt.x * rc.x + rt.y * rc.y + rt.z * rc.z;
      den += testCentered[k].x * testCentered[k].x +
        testCentered[k].y * testCentered[k].y +
        testCentered[k].z * testCentered[k].z;
    }
    if (den > 1e-12) scale = num / den;
    if (!Number.isFinite(scale) || scale <= 1e-9) scale = 1;
  }

  // Transform EVERY test point (valid or not) into the aligned ref frame so the
  // returned pose stays index-aligned with the source keypoint list.
  const points: Point3[] = test.map((p) => {
    if (!p || !p.valid) {
      return { x: 0, y: 0, z: 0, valid: false };
    }
    const centered = { x: p.x - testC.x, y: p.y - testC.y, z: p.z - testC.z };
    const rotated = applyMat(R, centered);
    return {
      x: rotated.x * scale,
      y: rotated.y * scale,
      z: rotated.z * scale,
      valid: true,
    };
  });

  return { points, rotation: R, scale };
}
