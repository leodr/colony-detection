export function getBase64FromObjectUrl(url: string) {
  return new Promise<string | ArrayBuffer | null>((resolve, reject) => {
    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const fileReader = new FileReader();
        fileReader.onload = () => {
          const base64 = fileReader.result;

          resolve(base64);
        };
        fileReader.onerror = reject;
        fileReader.readAsDataURL(blob);
      });
  });
}
