import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowUturnLeftIcon,
  EyeSlashIcon,
  TrashIcon,
} from "@heroicons/react/20/solid";
import {
  MouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDispatch, useSelector } from "react-redux";
import { tinykeys } from "tinykeys";
import { removeSelectedImage, selectPreviousImage } from "./images-slice";
import { mergeMaskImagesToImageElement } from "./merge-mask-images";
import { RootState } from "./store";
import { useStack } from "./use-stack";
import { Coordinates, getImageCoordinates } from "./util/image-coordinates";

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

  const {
    clear: clearClicks,
    pop: popClick,
    push: pushClick,
    stack: clickStack,
  } = useStack<Coordinates>(Infinity);

  const undo = useCallback(() => {
    popMask();
    popClick();
    setCounter((counter) => Math.max(counter - 1, 0));
  }, [popClick, popMask]);

  const clear = useCallback(() => {
    clearMasks();
    clearClicks();
    setCounter(0);
  }, [clearClicks, clearMasks]);

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
        undo();
      },
    });

    return () => {
      unsubscribe();
    };
  }, [dispatch, handleDownloadAndNext, undo]);

  useEffect(() => {
    if (selectedImage == null) {
      setImage(null);
      return;
    }

    clear();

    const imageUrl = images[selectedImage]?.url;

    if (imageUrl != null) {
      const img = new Image();
      img.onload = () => {
        setImage(img);
      };
      img.src = imageUrl;
    }
  }, [clear, images, selectedImage]);

  useEffect(() => {
    if (selectedImageUrl) {
      setImageSetState("loading");

      fetch(selectedImageUrl)
        .then((response) => response.blob())
        .then((blob) => {
          const formData = new FormData();

          formData.append("file", blob);

          fetch("http://localhost:8000/set-image", {
            method: "PUT",
            body: formData,
          }).finally(() => setImageSetState("idle"));
        });
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

      fetch(`http://localhost:8000/register-click`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: imageCoordinates.x,
          y: imageCoordinates.y,
        }),
      })
        .then((response) => response.blob())
        .then((blob) => {
          const img = new Image();
          img.onload = () => {
            pushMask(img);
          };
          img.src = URL.createObjectURL(blob);
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

      fetch("http://localhost:8000/colorize-from-coordinates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          coordinates: clickStack.map((e) => [e.x, e.y]),
          base64_image: canvas.toDataURL("image/png"),
        }),
      });

      // const link = document.createElement("a");

      // link.download =
      //   images[selectedImage]?.filename.split(".").slice(0, -1).join(".") +
      //   `_mask_${counter}.png`;
      // link.href = canvas.toDataURL("image/png");

      // link.click();
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
          onClick={undo}
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
          onClick={clear}
        >
          <TrashIcon
            className="-ml-0.5 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
          Clear mask
        </button>
        <div className="flex px-4">
          <div className="flex flex-col items-center gap-1">
            <span className="text-lg text-center leading-none">
              {new Intl.NumberFormat().format(counter)}
            </span>
            <span className="text-sm text-gray-500 leading-none">counted</span>
          </div>
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
