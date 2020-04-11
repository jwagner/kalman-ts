import Matrix from 'ml-matrix';
import KalmanFilter from '.';

test('constructor', () => {
  const subject = new KalmanFilter([0], new Matrix(1, 1));
  expect(subject.state.to2DArray()).toEqual([[0]]);
  expect(subject.stateCovariance.to2DArray()).toEqual([[0]]);
});

test('predict', () => {
  const subject = new KalmanFilter([2, 1], Matrix.eye(2, 2));
  const stateTransition = new Matrix([
    [1, 0.1],
    [0, 1],
  ]);
  const processNoise = Matrix.eye(2, 2, 1);
  subject.predict(stateTransition, processNoise);
  expect(subject.state.to2DArray()).toEqual([[2.1], [1]]);
  expect(subject.stateCovariance.to2DArray()).toEqual([
    [2.01, 0.1],
    [0.1, 2],
  ]);
});

test('update', () => {
  const subject = new KalmanFilter([2, 1], Matrix.eye(2, 2));
  subject.update([3], new Matrix([[1, 0]]), new Matrix([[1]]));
  expect(subject.state.to2DArray()).toEqual([[2.5], [1]]);
  expect(subject.stateCovariance.to2DArray()).toEqual([
    [0.5, 0],
    [0, 1],
  ]);
});
