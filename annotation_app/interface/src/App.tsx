import { Canvas } from "./Canvas";
import { ImageDropArea } from "./ImageDropArea";

function App() {
  return (
    <div className="p-3 flex min-h-screen">
      <nav className="w-96">
        <ImageDropArea></ImageDropArea>
      </nav>
      <Canvas></Canvas>
    </div>
  );
}

export default App;
