import { nanoid } from "nanoid";
import { DragEvent, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { addManyImages, selectImage } from "./images-slice";
import { RootState } from "./store";

export function ImageDropArea() {
  const dispatch = useDispatch();

  const imageIds = useSelector((state: RootState) => state.images.ids);
  const imageEntities = useSelector(
    (state: RootState) => state.images.entities
  );
  const selectedImage = useSelector(
    (state: RootState) => state.images.selectedImage
  );

  const [dragging, setDragging] = useState(false);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const images = [];

      for (const file of files) {
        const id = nanoid();
        const filename = file.name;
        const url = URL.createObjectURL(file);

        images.push({ id, filename, url });
      }

      dispatch(addManyImages(images));
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="h-full flex flex-col rounded-lg bg-gray-900 border-2 border-gray-700 p-2"
      style={{ borderColor: dragging ? "white" : undefined }}
    >
      {imageIds.map((id) => {
        const image = imageEntities[id];

        const isSelected = selectedImage === id;

        return (
          <button
            key={image.id}
            onClick={() => {
              dispatch(selectImage(id));
            }}
            className={`text-start text-sm py-2 px-2 flex gap-2 items-center rounded-md ${
              isSelected ? "bg-gray-700" : ""
            } ${isSelected ? "text-gray-200" : "text-gray-500"}`}
          >
            <img src={image.url} className="w-10 h-10 rounded object-cover" />
            <span
              className="break-all"
              style={{
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "normal",
              }}
            >
              {image.filename}
            </span>
          </button>
        );
      })}
    </div>
  );
}
