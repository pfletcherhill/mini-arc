import { Grid, COLORS } from "@/types";
import React, { useState } from "react";

interface GridProps {
  grid: Grid;
  paintColor: number;
  setGrid: (grid: Grid) => void;
}

const GridEditable: React.FC<GridProps> = ({ grid, paintColor, setGrid }) => {
  const [numRows, setNumRows] = useState<number>(grid.length);
  const [numCols, setNumCols] = useState<number>(grid[0].length);

  const handleResizeGrid = () => {
    const newGrid = [];
    for (let rowIndex = 0; rowIndex < numRows; rowIndex++) {
      const newRow: number[] = [];
      for (let colIndex = 0; colIndex < numCols; colIndex++) {
        newRow.push((grid[rowIndex] || [])[colIndex] || 0);
      }
      newGrid.push(newRow);
    }
    setGrid(newGrid);
  };

  const handleCellClick = (row: number, col: number) => {
    const newGrid = [...grid];
    newGrid[row] = [...newGrid[row]];
    newGrid[row][col] = paintColor;
    setGrid(newGrid);
  };

  return (
    <>
      <div className="inline-block border border-gray-500">
        {grid.map((row, rowIndex) => (
          <div key={rowIndex} className="flex">
            {row.map((cell, cellIndex) => (
              <div
                onClick={() => handleCellClick(rowIndex, cellIndex)}
                key={cellIndex}
                className="w-6 h-6 border-0.5 border-gray-500"
                style={{
                  backgroundColor: COLORS[cell],
                }}
              />
            ))}
          </div>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleResizeGrid();
        }}
      >
        Rows:{" "}
        <input
          value={numRows}
          onChange={(e) => {
            e.preventDefault();
            setNumRows(parseInt(e.target.value));
          }}
        />
        Columns:{" "}
        <input
          value={numCols}
          onChange={(e) => {
            e.preventDefault();
            setNumCols(parseInt(e.target.value));
          }}
        />
        <button type="submit">Resize</button>
      </form>
    </>
  );
};

export default GridEditable;
