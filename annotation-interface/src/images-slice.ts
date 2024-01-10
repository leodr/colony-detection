import {
  PayloadAction,
  createEntityAdapter,
  createSlice,
} from "@reduxjs/toolkit";

type ImageFromFile = { id: string; filename: string; url: string };

const imagesAdapter = createEntityAdapter({
  selectId: (image: ImageFromFile) => image.id,
  sortComparer: (a: ImageFromFile, b: ImageFromFile) =>
    a.filename.localeCompare(b.filename),
});

export interface ImagesState
  extends ReturnType<typeof imagesAdapter.getInitialState> {
  selectedImage: string | null;
}

const initialState: ImagesState = {
  ...imagesAdapter.getInitialState(),
  selectedImage: null,
};

const imagesSlice = createSlice({
  name: "images",
  initialState: initialState,
  reducers: {
    addImage: imagesAdapter.addOne,
    removeSelectedImage: (state) => {
      const currentlySelectedImage = state.selectedImage;

      if (currentlySelectedImage) {
        const index = state.ids.indexOf(currentlySelectedImage);
        const newId = state.ids[index + 1] || state.ids[0] || null;

        state.selectedImage = newId;

        URL.revokeObjectURL(state.entities[currentlySelectedImage]?.url);

        imagesAdapter.removeOne(state, currentlySelectedImage);
      }
    },
    selectImage: (state, action: PayloadAction<string>) => {
      state.selectedImage = action.payload;
    },
    addManyImages: imagesAdapter.addMany,
    selectPreviousImage: (state) => {
      if (!state.selectedImage) {
        state.selectedImage = state.ids[0] || null;
      } else {
        const index = state.ids.indexOf(state.selectedImage);
        const newId = state.ids[index - 1] || state.ids[0] || null;

        state.selectedImage = newId;
      }
    },
    selectNextImage: (state) => {
      if (!state.selectedImage) {
        state.selectedImage = state.ids[0] || null;
      } else {
        const index = state.ids.indexOf(state.selectedImage);
        const newId = state.ids[index + 1] || state.ids[0] || null;

        state.selectedImage = newId;
      }
    },
  },
});

export const {
  addImage,
  addManyImages,
  selectImage,
  selectNextImage,
  selectPreviousImage,
  removeSelectedImage,
} = imagesSlice.actions;

export const imagesReducer = imagesSlice.reducer;
