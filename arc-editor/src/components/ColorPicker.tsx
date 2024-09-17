import { COLORS } from "@/types";

interface ColorPickerProps {
  color: number;
  setColor: (color: number) => void;
}

const ColorPicker: React.FC<ColorPickerProps> = ({ color, setColor }) => {
  return (
    <>
      <div className="flex flex-row">
        {COLORS.map((colorHex, index) => (
          <div
            key={index}
            className={`w-10 h-10 cursor-pointer ${
              index == color
                ? "border-4 border-blue-500"
                : "border-0.5 border-gray-500"
            }`}
            style={{
              backgroundColor: colorHex,
            }}
            onClick={() => setColor(index)}
          >
            {index}
          </div>
        ))}
      </div>
    </>
  );
};

export default ColorPicker;
