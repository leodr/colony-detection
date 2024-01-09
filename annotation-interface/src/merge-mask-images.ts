export function mergeMaskImagesToImageElement(
  img1: HTMLImageElement,
  img2: HTMLImageElement
): Promise<HTMLImageElement> {
  // Create canvas and get context
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;

  // Set canvas size to accommodate the images
  canvas.width = Math.max(img1.width, img2.width);
  canvas.height = Math.max(img1.height, img2.height);

  // Draw the first image and get its data
  ctx.drawImage(img1, 0, 0);
  const imgData1 = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Draw the second image and get its data
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img2, 0, 0);
  const imgData2 = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Create new image data for the merged mask
  const mergedData = ctx.createImageData(canvas.width, canvas.height);

  // Merge the masks
  for (let i = 0; i < imgData1.data.length; i += 4) {
    if (imgData1.data[i] === 255 || imgData2.data[i] === 255) {
      // Set pixel to white if either corresponding pixel is white
      mergedData.data[i] = 255; // Red
      mergedData.data[i + 1] = 255; // Green
      mergedData.data[i + 2] = 255; // Blue
      mergedData.data[i + 3] = 255; // Alpha
    } else {
      // Set pixel to black otherwise
      mergedData.data[i] = 0;
      mergedData.data[i + 1] = 0;
      mergedData.data[i + 2] = 0;
      mergedData.data[i + 3] = 255;
    }
  }

  // Apply the merged mask to the canvas
  ctx.putImageData(mergedData, 0, 0);

  // Create a new image element
  const mergedImage = new Image();
  mergedImage.src = canvas.toDataURL();

  return new Promise((resolve) => {
    mergedImage.onload = () => resolve(mergedImage);
  });
}
