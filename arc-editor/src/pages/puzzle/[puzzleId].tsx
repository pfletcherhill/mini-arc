import { GetStaticProps, GetStaticPaths, NextPage } from "next";
import { useState } from "react";

import path from "path";
import fs from "fs";

import { Grid, Puzzle } from "@/types";
import GridStatic from "@/components/GridStatic";
import GridEditable from "@/components/GridEditable";

import ColorPicker from "@/components/ColorPicker";

interface PuzzlePageProps {
  puzzle: Puzzle;
  puzzleId: string;
}

const getMaxDim = (puzzle: Puzzle): number => {
  const dims: number[] = [];
  [...puzzle.test, ...puzzle.train].forEach(({ input, output }) => {
    dims.push(input.length, input[0].length, output.length, output[0].length);
  });
  return Math.max(...dims);
};

const PuzzlePage: NextPage<PuzzlePageProps> = ({ puzzle, puzzleId }) => {
  console.log(puzzle);
  const maxDim = getMaxDim(puzzle);
  const numPairs = puzzle.train.length;
  const [paintColor, setPaintColor] = useState<number>(0);
  const [editablePuzzle, setEditablePuzzle] = useState<Puzzle>(puzzle);

  const handleSetGrid = (
    set: "train" | "test",
    gridIndex: number,
    pair: "input" | "output",
    grid: Grid
  ): void => {
    const newPuzzle = { ...editablePuzzle };
    newPuzzle[set] = [...newPuzzle[set]];
    newPuzzle[set][gridIndex] = { ...newPuzzle[set][gridIndex] };
    newPuzzle[set][gridIndex][pair] = grid;
    setEditablePuzzle(newPuzzle);
  };

  const handleReset = (): void => {
    setEditablePuzzle({ ...puzzle });
  };

  return (
    <div>
      <div>
        <h1 className="text-lg font-bold">Puzzle ID: {puzzleId}</h1>
        <h3>Max dimension: {maxDim}</h3>
        <h3>Number of train pairs: {numPairs}</h3>
      </div>
      <div className="flex flex-row">
        <div className="w-1/2">
          <div>Original</div>

          <h3 className="text-xl font-semibold mb-4">Train Set</h3>
          {puzzle.train.map((grid, index) => (
            <div key={index} className="flex mb-4">
              <div className="mr-8">
                <h4 className="text-lg font-medium mb-2">Input</h4>
                <GridStatic grid={grid.input} />
              </div>
              <div>
                <h4 className="text-lg font-medium mb-2">Output</h4>
                <GridStatic grid={grid.output} />
              </div>
            </div>
          ))}

          <h3 className="text-xl font-semibold mb-4">Test Set</h3>
          {puzzle.test.map((grid, index) => (
            <div key={index} className="flex mb-4">
              <div className="mr-8">
                <h4 className="text-lg font-medium mb-2">Input</h4>
                <GridStatic grid={grid.input} />
              </div>
              <div>
                <h4 className="text-lg font-medium mb-2">Output</h4>
                <GridStatic grid={grid.output} />
              </div>
            </div>
          ))}
        </div>
        <div className="w-1/2">
          <div>Editable</div>

          <ColorPicker color={paintColor} setColor={setPaintColor} />

          <button
            onClick={(e) => {
              e.preventDefault();
              handleReset();
            }}
          >
            Reset to original
          </button>

          <h3 className="text-xl font-semibold mb-4">Train Set</h3>
          {editablePuzzle.train.map((grid, index) => (
            <div key={index} className="flex mb-4">
              <div className="mr-8">
                <h4 className="text-lg font-medium mb-2">Input</h4>
                <GridEditable
                  grid={grid.input}
                  paintColor={paintColor}
                  setGrid={(newGrid: Grid) =>
                    handleSetGrid("train", index, "input", newGrid)
                  }
                />
              </div>
              <div>
                <h4 className="text-lg font-medium mb-2">Output</h4>
                <GridEditable
                  grid={grid.output}
                  paintColor={paintColor}
                  setGrid={(newGrid: Grid) =>
                    handleSetGrid("train", index, "output", newGrid)
                  }
                />
              </div>
            </div>
          ))}

          <h3 className="text-xl font-semibold mb-4">Test Set</h3>
          {editablePuzzle.test.map((grid, index) => (
            <div key={index} className="flex mb-4">
              <div className="mr-8">
                <h4 className="text-lg font-medium mb-2">Input</h4>
                <GridEditable
                  grid={grid.input}
                  paintColor={paintColor}
                  setGrid={(newGrid: Grid) =>
                    handleSetGrid("test", index, "input", newGrid)
                  }
                />
              </div>
              <div>
                <h4 className="text-lg font-medium mb-2">Output</h4>
                <GridEditable
                  grid={grid.output}
                  paintColor={paintColor}
                  setGrid={(newGrid: Grid) =>
                    handleSetGrid("test", index, "output", newGrid)
                  }
                />
              </div>
            </div>
          ))}

          <div>{JSON.stringify(editablePuzzle)}</div>
        </div>
      </div>
    </div>
  );
};

export const getStaticPaths: GetStaticPaths = async () => {
  const filePath = path.join(process.cwd(), "public", "combined.json");
  const jsonData = fs.readFileSync(filePath, "utf8");
  const Puzzles = JSON.parse(jsonData);

  const paths = Object.keys(Puzzles).map((puzzleId) => ({
    params: { puzzleId },
  }));

  return { paths, fallback: false };
};

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const puzzleId = params?.puzzleId as string;
  const filePath = path.join(process.cwd(), "public", "combined.json");
  const jsonData = fs.readFileSync(filePath, "utf8");
  const Puzzles = JSON.parse(jsonData);

  return {
    props: {
      puzzle: Puzzles[puzzleId],
      puzzleId,
    },
  };
};

export default PuzzlePage;
