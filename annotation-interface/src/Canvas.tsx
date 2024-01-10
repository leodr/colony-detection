import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowUturnLeftIcon,
  EyeSlashIcon,
  MinusIcon,
  PlusIcon,
  TrashIcon,
} from "@heroicons/react/20/solid";
import { MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { tinykeys } from "tinykeys";
import { removeSelectedImage, selectPreviousImage } from "./images-slice";
import { mergeMaskImagesToImageElement } from "./merge-mask-images";
import { RootState } from "./store";
import { useStack } from "./use-stack";
import { getBase64FromObjectUrl } from "./util/base64-from-objecturl";
import { getImageCoordinates } from "./util/image-coordinates";

export function Canvas() {
  const dispatch = useDispatch();
  const selectedImage = useSelector(
    (state: RootState) => state.images.selectedImage
  );
  const images = useSelector((state: RootState) => state.images.entities);

  const [imageSetState, setImageSetState] = useState<string>("idle");

  const selectedImageUrl = useMemo(() => {
    if (selectedImage == null) {
      return null;
    }

    return images[selectedImage]?.url;
  }, [images, selectedImage]);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const {
    top: mask,
    clear: clearMasks,
    push: pushMask,
    pop: popMask,
  } = useStack<HTMLImageElement>(5);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  const [showMask, setShowMask] = useState(true);
  const [counter, setCounter] = useState(0);

  useEffect(() => {
    const unsubscribe = tinykeys(window, {
      h: () => {
        setShowMask((curr) => !curr);
      },
      ArrowLeft: () => {
        dispatch(selectPreviousImage());
      },
      ArrowRight: () => {
        handleDownloadAndNext();
      },
      "+": () => {
        setCounter((counter) => counter + 1);
      },
      "-": () => {
        setCounter((counter) => Math.max(counter - 1, 0));
      },
      "$mod+z": () => {
        popMask();
        setCounter((counter) => Math.max(counter - 1, 0));
      },
    });

    return () => {
      unsubscribe();
    };
  }, [dispatch, handleDownloadAndNext, popMask]);

  useEffect(() => {
    if (selectedImage == null) {
      setImage(null);
      return;
    }

    clearMasks();
    setCounter(0);

    const imageUrl = images[selectedImage]?.url;

    if (imageUrl != null) {
      const img = new Image();
      img.onload = () => {
        setImage(img);
      };
      img.src = imageUrl;
    }
  }, [clearMasks, images, selectedImage]);

  useEffect(() => {
    if (selectedImageUrl) {
      setImageSetState("loading");
      getBase64FromObjectUrl(selectedImageUrl)
        .then((base64) =>
          fetch("http://localhost:8000/set-image", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base64_image: base64 }),
          })
        )
        .finally(() => setImageSetState("idle"));
    }
  }, [selectedImageUrl]);

  useEffect(() => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d")!;
      const canvasWidth = canvasRef.current.width;
      const canvasHeight = canvasRef.current.height;

      if (image) {
        ctx.drawImage(image, 0, 0, canvasWidth, canvasHeight);

        if (mask && showMask) {
          const canvas = document.createElement("canvas");
          const ctx2 = canvas.getContext("2d")!;

          canvas.width = canvasWidth;
          canvas.height = canvasHeight;

          ctx2.drawImage(mask, 0, 0, canvasWidth, canvasHeight);
          const imageData = ctx2.getImageData(0, 0, canvasWidth, canvasHeight);

          for (let i = 0; i < imageData.data.length; i += 4) {
            // mergedData.data[i] = 255; // Red
            // mergedData.data[i + 1] = 255; // Green
            // mergedData.data[i + 2] = 255; // Blue
            imageData.data[i + 3] = 128; // Alpha
          }

          ctx2.putImageData(imageData, 0, 0);

          ctx.drawImage(canvas, 0, 0, canvasWidth, canvasHeight);
        }
      } else {
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      }
    }
  }, [image, mask, showMask]);

  function handleCanvasClick(e: MouseEvent<HTMLCanvasElement>) {
    if (image != null && imageSetState === "idle") {
      if (!e.altKey) {
        setCounter((counter) => counter + 1);
      }

      const relativeX = e.nativeEvent.offsetX;
      const relativeY = e.nativeEvent.offsetY;

      const target = e.target as HTMLCanvasElement;

      // Calculate screen size of canvas
      const canvasWidth = target.clientWidth;
      const canvasHeight = target.clientHeight;

      const imageCoordinates = getImageCoordinates(
        { x: relativeX, y: relativeY },
        { width: canvasWidth, height: canvasHeight },
        { width: image.width, height: image.height }
      );

      fetch(
        `http://localhost:8000/predict-from-coords?x=${imageCoordinates.x}&y=${imageCoordinates.y}`
      ).then((response) => {
        response.json().then((json) => {
          const image = json["mask_image"];

          // Read image as base64 and set it as the mask
          const img = new Image();
          img.onload = () => {
            if (mask) {
              mergeMaskImagesToImageElement(mask, img).then((mergedImage) => {
                pushMask(mergedImage);
              });
            } else {
              pushMask(img);
            }
          };
          // Add base64 prefix
          img.src = "data:image/png;base64," + image;
        });
      });
    }
  }

  function handleDownloadAndNext() {
    if (image && selectedImage) {
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;

      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, image.width, image.height);
      if (mask) {
        ctx.drawImage(mask, 0, 0, mask.width, mask.height);
      }

      const link = document.createElement("a");

      link.download =
        images[selectedImage]?.filename.split(".").slice(0, -1).join(".") +
        `_mask_${counter}.png`;
      link.href = canvas.toDataURL("image/png");

      link.click();
    }

    dispatch(removeSelectedImage());
  }

  return (
    <main className="flex flex-1 items-center justify-center relative">
      <div className="relative">
        <canvas
          className={`rounded border border-gray-500 w-[800px] h-[800px] graph ${
            imageSetState === "loading" ? "animate-pulse grayscale" : ""
          }`}
          ref={canvasRef}
          width={image?.width ?? 800}
          height={image?.height ?? 800}
          onClick={handleCanvasClick}
        />
        <button
          type="button"
          className="absolute top-100 mt-2 left-1/2 transform -translate-x-1/2 inline-flex items-center gap-x-2 rounded-sm px-3 py-1.5 text-sm text-gray-200 shadow-sm hover:bg-white/10 border border-gray-700"
          onClick={() => {
            setShowMask((curr) => !curr);
          }}
        >
          <EyeSlashIcon
            className="-ml-0.5 h-4 w-4 text-gray-500"
            aria-hidden="true"
          />
          <div className="grid">
            <span
              className="col-start-1 col-span-1 row-start-1 row-span-1"
              aria-hidden={showMask}
              style={{ visibility: !showMask ? "hidden" : "visible" }}
            >
              Hide mask
            </span>
            <span
              className="col-start-1 col-span-1 row-start-1 row-span-1"
              aria-hidden={!showMask}
              style={{ visibility: showMask ? "hidden" : "visible" }}
            >
              Show mask
            </span>
          </div>
        </button>
      </div>
      <div className="flex whitespace-nowrap gap-2 items-center shadow absolute text-white bottom-0 left-1/2 transform -translate-x-1/2 bg-gray-800 p-2 border-2 border-gray-500 rounded-lg">
        <button
          type="button"
          className="inline-flex items-center gap-x-2 rounded-md bg-white/10 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-white/20"
          onClick={() => {
            popMask();
            setCounter((counter) => Math.max(counter - 1, 0));
          }}
        >
          <ArrowUturnLeftIcon
            className="-ml-0.5 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
          Undo
        </button>
        <button
          type="button"
          className="inline-flex items-center gap-x-2 rounded-md bg-white/10 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-white/20"
          onClick={() => {
            clearMasks();
            setCounter(0);
          }}
        >
          <TrashIcon
            className="-ml-0.5 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
          Clear mask
        </button>
        <div className="flex px-4">
          <>
            <button
              className="p-2 text-gray-400 hover:text-gray-200"
              onClick={(e) => {
                const decrease = e.shiftKey ? 10 : 1;
                setCounter((counter) => Math.max(counter - decrease, 0));
              }}
            >
              <MinusIcon className="-ml-0.5 h-5 w-5" aria-hidden="true" />
            </button>
          </>
          <div className="flex flex-col items-center gap-1">
            <span className="text-lg text-center leading-none">
              {new Intl.NumberFormat().format(counter)}
            </span>
            <span className="text-sm text-gray-500 leading-none">counted</span>
          </div>
          <button
            className="p-2 text-gray-400 hover:text-gray-200"
            onClick={(e) => {
              const increase = e.shiftKey ? 10 : 1;
              setCounter((counter) => counter + increase);
            }}
          >
            <PlusIcon className="-ml-0.5 h-5 w-5" aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-x-2 rounded-md bg-white/10 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-white/20"
          onClick={() => {
            dispatch(selectPreviousImage());
          }}
        >
          <ArrowLeftIcon
            className="-ml-0.5 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
        </button>
        <button
          type="button"
          className="inline-flex items-center gap-x-2 rounded-md bg-white/10 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-white/20"
          onClick={handleDownloadAndNext}
        >
          <ArrowRightIcon
            className="-ml-0.5 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
          Download & Next
        </button>
      </div>
    </main>
  );
}
