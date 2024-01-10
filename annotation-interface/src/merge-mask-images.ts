export function mergeMaskImagesToImageElement(
  previousImage: HTMLImageElement,
  newImage: HTMLImageElement
): Promise<HTMLImageElement> {
  // Create canvas and get context
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;

  // Set canvas size to accommodate the images
  canvas.width = Math.max(previousImage.width, newImage.width);
  canvas.height = Math.max(previousImage.height, newImage.height);

  // Draw the first image and get its data
  ctx.drawImage(previousImage, 0, 0);
  const imgData1 = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Draw the second image and get its data
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(newImage, 0, 0);
  const imgData2 = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Merge the masks
  for (let i = 0; i < imgData1.data.length; i += 4) {
    if (imgData2.data[i] === 255) {
      // Set pixel to white if either corresponding pixel is white
      imgData1.data[i] = 255; // Red
      imgData1.data[i + 1] = 255; // Green
      imgData1.data[i + 2] = 255; // Blue
      imgData1.data[i + 3] = 255; // Alpha
    }
  }

  // Apply the merged mask to the canvas
  ctx.putImageData(imgData1, 0, 0);

  // Create a new image element
  const mergedImage = new Image();
  mergedImage.src = canvas.toDataURL();

  return new Promise((resolve) => {
    mergedImage.onload = () => resolve(mergedImage);
  });
}
