export type Coordinates = {
  x: number;
  y: number;
};

export type Size = {
  width: number;
  height: number;
};

/**
 * Convert coordinates from the canvas to coordinates in the image
 */
export function getImageCoordinates(
  canvasCoordinates: Coordinates,
  canvasSize: Size,
  imageSize: Size
): Coordinates {
  const x = Math.round(
    (canvasCoordinates.x / canvasSize.width) * imageSize.width
  );
  const y = Math.round(
    (canvasCoordinates.y / canvasSize.height) * imageSize.height
  );
  return { x, y };
}
