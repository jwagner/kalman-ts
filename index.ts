import Matrix, { inverse } from 'ml-matrix';

export default class KalmanFilter {
  public state: Matrix;
  public stateCovariance: Matrix;

  constructor(initialState: number[], initialStateCovariance: Matrix) {
    this.state = new Matrix([initialState]).transpose();
    this.stateCovariance = initialStateCovariance;
  }

  predict(
    stateTransition: /* F */ Matrix,
    processNoise: /* Q */ Matrix,
    alpha: number = 1.0 // fading memory
  ) {
    this.state = stateTransition.mmul(this.state);
    this.stateCovariance = stateTransition
      .mmul(this.stateCovariance)
      .mmul(stateTransition.transpose())
      .mul(alpha * alpha)
      .add(processNoise);
  }

  update(
    measurements: /* z */ number[],
    measurementFunction: /* H */ Matrix,
    measurementNoise: /* R */ Matrix
  ) {
    const x = this.state;
    const P = this.stateCovariance;
    const z = new Matrix([measurements]).transpose();
    const H = measurementFunction;
    const Ht = H.transpose();
    const R = measurementNoise;
    const S = measurementFunction.mmul(P).mmul(Ht).add(R);
    const K = P.mmul(Ht).mmul(inverse(S));
    const y = z.sub(H.mmul(x));
    const I = Matrix.eye(x.rows, x.rows);
    this.state = x.add(K.mmul(y));

    const IsubKdotH = I.clone().sub(K.mmul(H));
    this.stateCovariance = IsubKdotH.mmul(P)
      .mmul(IsubKdotH.transpose())
      .add(K.mmul(R).mmul(K.transpose()));
  }

  getState() {
    return this.state.to1DArray();
  }
}

export function smoothRTS(
  state: /* x */ Matrix[],
  stateCovariance: /* P */ Matrix[],
  stateTransition: /* F */ Matrix[],
  processNoise: /*Q*/ Matrix[]
): [Matrix[], Matrix[]] {
  const dim = state[0].rows;
  const x = state.map((m) => m.clone());
  const P = stateCovariance.map((m) => m.clone());
  const Pp = P.map((m) => m.clone());

  const K = x.map(() => new Matrix(dim, dim));
  // smoothing
  for (let k = x.length - 2; k--; k >= 0) {
    const F = stateTransition[k];
    const Q = processNoise[k];
    const Ft = F.transpose();
    Pp[k] = F.mmul(P[k]).mmul(Ft).add(Q);
    K[k] = P[k].mmul(Ft).mmul(inverse(Pp[k]));
    x[k] = x[k].add(K[k].mmul(x[k + 1].clone().sub(F.mmul(x[k]))));
    P[k] = P[k]
      .add(K[k].mmul(P[k + 1].clone().sub(Pp[k])))
      .mmul(K[k].transpose());
  }
  return [x, P];
}
