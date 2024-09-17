export const COLORS = [
  // "#c2c0c0", // padding grey
  "#111111", // black
  "#1E93FF", // blue
  "#F93C31", // red
  "#4FCC30", // green
  "#FFDC00", // yellow
  "#E6E6E6", // grey
  "#E53AA3", // magenta
  "#FF851B", // orange
  "#87D8F1", // light blue
  "#921231", // maroon
  "#FFFFFF",
];

export type Grid = number[][];
export type GridPair = { input: Grid; output: Grid };
export type Puzzle = {
  test: GridPair[];
  train: GridPair[];
};
