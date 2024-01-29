import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowUturnLeftIcon,
  EyeSlashIcon,
  TrashIcon,
} from "@heroicons/react/20/solid";
import { MouseEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
// @ts-expect-error: No types are available
import { tinykeys } from "tinykeys";
import { removeSelectedImage, selectPreviousImage } from "./images-slice";
import { RootState } from "./store";

type Quadrant =
  | "top-left"
  | "top-right"
  | "bottom-left"
  | "bottom-right"
  | "all";

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

  const [mask, setMask] = useState<HTMLImageElement | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  const [showMask, setShowMask] = useState(true);
  const [counter, setCounter] = useState(0);

  const [quadrant, setQuadrant] = useState<Quadrant>("all");

  const undo = useCallback(() => {
    fetch("http://localhost:8000/undo", { method: "POST" })
      .then((response) => response.blob())
      .then((blob) => {
        const img = new Image();
        img.onload = () => {
          setMask(img);
        };
        img.src = URL.createObjectURL(blob);
      });
  }, []);

  const clear = useCallback(() => {
    fetch("http://localhost:8000/clear-mask", { method: "POST" })
      .then((response) => response.blob())
      .then((blob) => {
        const img = new Image();
        img.onload = () => {
          setMask(img);
        };
        img.src = URL.createObjectURL(blob);
      });
  }, []);

  useEffect(() => {
    if (mask) {
      fetch("http://localhost:8000/click-count")
        .then((response) => response.text())
        .then((text) => setCounter(Number(text)));
    } else {
      setCounter(0);
    }
  }, [mask]);

  const handleDownloadAndNext = useCallback(() => {
    if (image && selectedImage && mask) {
      const link = document.createElement("a");

      link.download =
        images[selectedImage]?.filename.split(".").slice(0, -1).join(".") +
        `_mask_${counter}.png`;
      link.href = mask.src;

      link.click();
    }

    dispatch(removeSelectedImage());
  }, [counter, dispatch, image, images, mask, selectedImage]);

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

  function handleMaskClick(e: MouseEvent<HTMLImageElement>) {
    if (image != null && imageSetState === "idle") {
      const relativeX = e.nativeEvent.offsetX;
      const relativeY = e.nativeEvent.offsetY;

      const target = e.target as HTMLImageElement;

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
            setMask(img);
          };
          img.src = URL.createObjectURL(blob);
        });
    }
  }

  const positioningClasses = ["max-w-[unset] absolute"];

  switch (quadrant) {
    case "top-left":
      positioningClasses.push("w-[200%] h-[200%] top-0 left-0");
      break;
    case "top-right":
      positioningClasses.push("w-[200%] h-[200%] top-0 -left-full");
      break;
    case "bottom-left":
      positioningClasses.push("w-[200%] h-[200%] -top-full left-0");
      break;
    case "bottom-right":
      positioningClasses.push("w-[200%] h-[200%] -top-full -left-full");
      break;
    case "all":
      positioningClasses.push("w-full h-full top-0 left-0");
      break;
  }

  const positioningClassesString = positioningClasses.join(" ");

  return (
    <main className="flex flex-1 items-center justify-center relative">
      <div className="relative">
        <div
          className={`relative rounded border-2 border-gray-700 w-[800px] h-[800px] graph overflow-hidden ${
            imageSetState === "loading" ? "animate-pulse grayscale" : ""
          }`}
        >
          {selectedImageUrl ? (
            <img src={selectedImageUrl} className={positioningClassesString} />
          ) : null}
          {mask && showMask ? (
            <img
              src={mask.src}
              alt=""
              className={positioningClassesString + " opacity-50 transform-gpu"}
              onClick={handleMaskClick}
            />
          ) : null}
        </div>
        <div className="absolute top-100 mt-2 left-1/2 transform -translate-x-1/2 flex gap-x-2">
          <button
            type="button"
            className="inline-flex items-center gap-x-2 rounded-sm px-3 py-1.5 text-sm text-gray-200 shadow-sm hover:bg-white/10 border border-gray-700"
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
          <div className="flex items-center gap-x-1 rounded-sm px-2 py-1 text-sm shadow-sm border border-gray-700">
            <button
              onClick={() => setQuadrant("all")}
              className={
                "p-1 rounded " +
                (quadrant === "all" ? "bg-white/20" : "hover:bg-white/10")
              }
            >
              <div className="rounded-full w-4 h-4 bg-gray-500"></div>
            </button>
            <button
              onClick={() => setQuadrant("top-left")}
              className={
                "p-1 rounded " +
                (quadrant === "top-left" ? "bg-white/20" : "hover:bg-white/10")
              }
            >
              <div className="rounded-tl-full w-4 h-4 bg-gray-500"></div>
            </button>
            <button
              onClick={() => setQuadrant("top-right")}
              className={
                "p-1 rounded " +
                (quadrant === "top-right" ? "bg-white/20" : "hover:bg-white/10")
              }
            >
              <div className="rounded-tr-full w-4 h-4 bg-gray-500"></div>
            </button>
            <button
              onClick={() => setQuadrant("bottom-left")}
              className={
                "p-1 rounded " +
                (quadrant === "bottom-left"
                  ? "bg-white/20"
                  : "hover:bg-white/10")
              }
            >
              <div className="rounded-bl-full w-4 h-4 bg-gray-500"></div>
            </button>
            <button
              onClick={() => setQuadrant("bottom-right")}
              className={
                "p-1 rounded " +
                (quadrant === "bottom-right"
                  ? "bg-white/20"
                  : "hover:bg-white/10")
              }
            >
              <div className="rounded-br-full w-4 h-4 bg-gray-500"></div>
            </button>
          </div>
        </div>
      </div>
      <div className="flex whitespace-nowrap gap-2 items-center shadow absolute text-white bottom-0 left-1/2 transform -translate-x-1/2 bg-gray-900 p-2 border-2 border-gray-700 rounded-lg">
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
function getImageCoordinates(
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
